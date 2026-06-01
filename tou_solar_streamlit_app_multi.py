import io
import re
import zipfile
from threading import RLock

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

# Matplotlib can be unsafe with concurrent Streamlit sessions.
_PLOT_LOCK = RLock()

LABEL_FMT = "%a, %d-%b-%Y %H:%M"


# -----------------------------
# Helper functions
# -----------------------------
def extract_cabin_name(filename: str) -> str:
    """Extract a clean cabin name such as P3415 or PT1252 from uploaded filename."""
    stem = filename.rsplit(".", 1)[0]
    match = re.search(r"\b[A-Za-z]{1,5}\d{2,8}\b", stem)
    if match:
        return match.group(0).upper()
    return stem.split("_")[0].split("-")[0].upper()


def safe_filename(text: str) -> str:
    """Make a safe filename for exported files."""
    text = str(text).strip()
    text = re.sub(r"[^\w\-]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "output"


def extract_numeric(value):
    """Extract first numeric value from values such as '171.841 kW'."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


@st.cache_data(show_spinner=False)
def read_csv_flexible_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV from bytes with automatic separator detection and fallback."""
    try:
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes))


def guess_column(columns, candidates):
    """Return best matching column name from candidate keywords."""
    if len(columns) == 0:
        return None

    normalized = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]

    for col in columns:
        col_l = str(col).strip().lower()
        if any(c.strip().lower() in col_l for c in candidates):
            return col

    return columns[0]


def guess_datetime_col(columns):
    return guess_column(columns, ["Date/Time", "DateTime", "Timestamp", "Date", "Time"])


def guess_value_col(columns):
    return guess_column(columns, ["Value", "Watt Total Avg", "Watt Total  Avg", "kW", "KW", "Active Power", "Power"])


def guess_unit_mode(value_col: str) -> str:
    """Guess whether value column is in W or already kW."""
    col_l = str(value_col).lower()
    if "watt" in col_l and "kw" not in col_l:
        return "W → kW"
    return "Already kW"


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse datetime robustly."""
    known_formats = [
        "%d-%b-%y %I:%M:%S %p",
        "%d-%b-%Y %I:%M:%S %p",
        "%d-%b-%y %H:%M:%S",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    best = None
    best_valid = -1

    for fmt in known_formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best = parsed
            best_valid = valid

    fallback_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
    fallback_valid = int(fallback_dayfirst.notna().sum())
    if fallback_valid > best_valid:
        return fallback_dayfirst

    return best


def convert_to_kw(series: pd.Series, unit_mode: str) -> pd.Series:
    """Convert selected value column to kW."""
    numeric = series.apply(extract_numeric).astype(float)
    if unit_mode == "W → kW":
        return numeric / 1000.0
    return numeric


def process_daily_max(df: pd.DataFrame, datetime_col: str, value_col: str, unit_mode: str) -> pd.DataFrame:
    """Return one row per day: timestamp of daily peak and daily max kW."""
    timestamps = parse_datetime_series(df[datetime_col])
    kw = convert_to_kw(df[value_col], unit_mode)

    data = (
        pd.DataFrame({"timestamp": timestamps, "kW": kw})
        .dropna(subset=["timestamp", "kW"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    if data.empty:
        return pd.DataFrame()

    grouped = data["kW"].groupby(data.index.date)
    idx_of_max = grouped.idxmax()
    val_of_max = grouped.max()

    result = pd.DataFrame(
        {
            "peak_timestamp": pd.to_datetime(idx_of_max.values),
            "daily_max_kW": val_of_max.values,
        },
        index=pd.to_datetime(idx_of_max.index),
    )
    result.index.name = "date"
    return result


def filter_by_date(result: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if result.empty:
        return result
    mask = (
        (result["peak_timestamp"].dt.date >= start_date)
        & (result["peak_timestamp"].dt.date <= end_date)
    )
    return result.loc[mask].copy()


def build_summary_table(filtered: pd.DataFrame, contract_kw: float) -> pd.DataFrame:
    """Build the clean daily table shown/exported to users."""
    table = pd.DataFrame(
        {
            "No.": range(1, len(filtered) + 1),
            "Date": filtered["peak_timestamp"].dt.strftime(LABEL_FMT),
            "Daily Max kW": filtered["daily_max_kW"].round(2),
            "Contract kW": round(contract_kw, 2),
        }
    )
    table["Over Contract"] = (table["Daily Max kW"] - table["Contract kW"]).clip(lower=0).round(2)
    return table


def adaptive_tick(step_candidates, vmin, vmax):
    rng = max(1e-9, vmax - vmin)
    for step in step_candidates:
        if rng / step <= 12:
            return step
    return step_candidates[-1]


def make_chart(filtered: pd.DataFrame, cabin_name: str, contract_kw: float, start_date, end_date):
    labels = filtered["peak_timestamp"].dt.strftime(LABEL_FMT).tolist()
    yvals = filtered["daily_max_kW"].to_numpy(dtype=float)
    xpos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(xpos, yvals, marker="o", label="Daily Max (kW)")
    ax.axhline(contract_kw, linestyle="--", label=f"Contract {contract_kw:g} kW")

    for xi, yi in zip(xpos, yvals):
        ax.vlines(
            xi,
            min(yi, contract_kw),
            max(yi, contract_kw),
            linestyles="--",
            alpha=0.35,
        )

    ymin = min(np.min(yvals), contract_kw)
    ymax = max(np.max(yvals), contract_kw)
    y_range = ymax - ymin if ymax > ymin else max(abs(ymax), 1.0)

    # Add enough vertical space so labels/annotation do not hit the title.
    bottom_pad = 0.08 * y_range
    top_pad = 0.22 * y_range
    ax.set_ylim(ymin - bottom_pad, ymax + top_pad)

    max_idx = int(np.argmax(yvals))
    max_x = xpos[max_idx]
    max_y = yvals[max_idx]
    max_label = labels[max_idx]

    # If the peak is close to the top of the axis, place the annotation below the point.
    # Otherwise, place it above. This prevents collision with the chart title.
    axis_top = ymax + top_pad
    top_distance_ratio = (axis_top - max_y) / (axis_top - (ymin - bottom_pad))

    if top_distance_ratio < 0.18:
        annotation_offset = (0, -45)
        annotation_va = "top"
    else:
        annotation_offset = (0, 32)
        annotation_va = "bottom"

    ax.annotate(
        f"Max: {max_y:.2f} kW\n{max_label}",
        xy=(max_x, max_y),
        xytext=annotation_offset,
        textcoords="offset points",
        ha="center",
        va=annotation_va,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="gray", alpha=0.85),
        annotation_clip=False,
    )

    ax.set_title(
        f"Daily Maximum Demand – {cabin_name}\n{start_date} to {end_date}",
        pad=22,
        weight="bold",
    )
    ax.set_ylabel("ACTIVE POWER (kW)")
    ax.set_xlabel("")

    ax.yaxis.set_major_locator(
        mticker.MultipleLocator(adaptive_tick([10, 20, 50, 100], ymin, ymax))
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="lower left", frameon=False)

    # Leave extra room for the title and rotated x-axis labels.
    fig.subplots_adjust(top=0.78, bottom=0.32)
    return fig



def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", transparent=True)
    buffer.seek(0)
    return buffer.getvalue()


def make_table_png(table: pd.DataFrame, title: str = "Daily Maximum Demand Table") -> bytes:
    highlight_color = "#fff2cc"
    display_df = table.copy()
    for col in ["Daily Max kW", "Contract kW", "Over Contract"]:
        display_df[col] = display_df[col].map(lambda v: f"{float(v):,.2f}")

    fig_h = max(3, len(display_df) * 0.35)
    fig, ax = plt.subplots(figsize=(11, fig_h + 1.0))
    ax.axis("off")
    ax.set_title(title, pad=12, weight="bold")

    col_widths = [0.25, 1.7, 0.9, 0.9, 0.95]
    norm_col_widths = [w / sum(col_widths) for w in col_widths]
    mpl_table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        colWidths=norm_col_widths,
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)
    mpl_table.scale(1, 1.2)

    for (row, col), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_linewidth(0.6)
            cell.set_edgecolor("black")
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_linewidth(0.3)
            cell.set_edgecolor("#cccccc")

    if len(table) > 0:
        idx_high = int(table["Daily Max kW"].astype(float).values.argmax()) + 1
        for col in range(display_df.shape[1]):
            mpl_table[(idx_high, col)].set_facecolor(highlight_color)

    fig.subplots_adjust(bottom=0)
    fig.patches.append(
        Rectangle((0.1, 0.04), 0.02, 0.02, transform=fig.transFigure, facecolor=highlight_color, edgecolor="gray")
    )
    fig.text(0.13, 0.05, "Highest Daily Max kW", va="center", fontsize=10)

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel_bytes(sheets: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = str(sheet_name)[:31] or "Sheet1"
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
    buffer.seek(0)
    return buffer.getvalue()


def build_file_records(uploaded_files):
    records = []
    used_labels = set()

    for idx, file in enumerate(uploaded_files, start=1):
        cabin = extract_cabin_name(file.name)
        base_label = f"{cabin} — {file.name}"
        label = base_label
        n = 2
        while label in used_labels:
            label = f"{base_label} ({n})"
            n += 1
        used_labels.add(label)

        records.append(
            {
                "label": label,
                "filename": file.name,
                "cabin": cabin,
                "bytes": file.getvalue(),
            }
        )
    return records


def analyze_record_auto(record: dict, contract_kw: float, start_date=None, end_date=None) -> dict:
    """Analyze one uploaded file using detected columns. Used for batch mode."""
    output = {
        "record": record,
        "contract_kw": contract_kw,
        "ok": False,
        "error": "",
        "raw_df": None,
        "result": pd.DataFrame(),
        "filtered": pd.DataFrame(),
        "table": pd.DataFrame(),
        "datetime_col": None,
        "value_col": None,
        "unit_mode": None,
    }

    try:
        raw_df = read_csv_flexible_from_bytes(record["bytes"])
        output["raw_df"] = raw_df

        if raw_df.empty:
            output["error"] = "CSV is empty."
            return output

        columns = list(raw_df.columns)
        datetime_col = guess_datetime_col(columns)
        value_col = guess_value_col(columns)
        unit_mode = guess_unit_mode(value_col)

        output["datetime_col"] = datetime_col
        output["value_col"] = value_col
        output["unit_mode"] = unit_mode

        if datetime_col is None or value_col is None:
            output["error"] = "Could not detect required columns."
            return output

        result = process_daily_max(raw_df, datetime_col, value_col, unit_mode)
        output["result"] = result

        if result.empty:
            output["error"] = "No valid timestamp/kW rows found."
            return output

        if start_date is None:
            start_date = result["peak_timestamp"].dt.date.min()
        if end_date is None:
            end_date = result["peak_timestamp"].dt.date.max()

        filtered = filter_by_date(result, start_date, end_date)
        output["filtered"] = filtered

        if filtered.empty:
            output["error"] = "No data found inside selected date range."
            return output

        table = build_summary_table(filtered, contract_kw)
        output["table"] = table
        output["ok"] = True
        return output

    except Exception as exc:
        output["error"] = str(exc)
        return output


def calculate_over_contract(filtered: pd.DataFrame, contract_kw: float):
    """Calculate over-contract statistics without adding extra columns to the daily table."""
    excess_series = (filtered["daily_max_kW"] - float(contract_kw)).clip(lower=0)
    exceed_days = int((excess_series > 0).sum())
    max_excess = float(excess_series.max()) if len(excess_series) else 0.0
    return exceed_days, max_excess

def summarize_analysis(analysis: dict, start_date, end_date) -> dict:
    record = analysis["record"]
    filtered = analysis["filtered"]
    contract_kw = analysis["contract_kw"]

    peak_row = filtered.loc[filtered["daily_max_kW"].idxmax()]
    peak_kw = float(peak_row["daily_max_kW"])
    peak_time = peak_row["peak_timestamp"]
    exceed_days, max_excess = calculate_over_contract(filtered, contract_kw)

    return {
        "Cabin": record["cabin"],
        "Filename": record["filename"],
        "Contract kW": round(float(contract_kw), 2),
        "Start Date": str(start_date),
        "End Date": str(end_date),
        "Maximum kW": round(peak_kw, 2),
        "Peak Time": peak_time.strftime(LABEL_FMT),
        "Days Over Contract": exceed_days,
        "Highest Excess kW": round(max_excess, 2),
        "Status": "Over Contract" if exceed_days > 0 else "OK",
        "Detected Date Column": analysis["datetime_col"],
        "Detected Value Column": analysis["value_col"],
        "Detected Unit": analysis["unit_mode"],
        "Rows Used": int(len(filtered)),
    }


def build_zip_bundle(successful_analyses: list, combined_summary: pd.DataFrame, combined_daily: pd.DataFrame, start_date, end_date) -> bytes:
    """Create one ZIP containing per-cabin charts/tables plus combined summaries."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("combined_summary.csv", dataframe_to_csv_bytes(combined_summary))
        zf.writestr("combined_daily_max.csv", dataframe_to_csv_bytes(combined_daily))

        excel_bytes = dataframe_to_excel_bytes(
            {
                "Combined Summary": combined_summary,
                "Combined Daily Max": combined_daily,
            }
        )
        zf.writestr("combined_summary.xlsx", excel_bytes)

        for analysis in successful_analyses:
            record = analysis["record"]
            cabin_safe = safe_filename(record["cabin"])
            table = analysis["table"]
            filtered = analysis["filtered"]
            contract_kw = analysis["contract_kw"]

            with _PLOT_LOCK:
                fig = make_chart(filtered, record["cabin"], contract_kw, start_date, end_date)
                chart_png = fig_to_png_bytes(fig)
                plt.close(fig)

            table_png = make_table_png(table, title=f"Daily Max Table – {record['cabin']}")

            zf.writestr(f"{cabin_safe}/{cabin_safe}_daily_max_chart.png", chart_png)
            zf.writestr(f"{cabin_safe}/{cabin_safe}_daily_max_table.png", table_png)
            zf.writestr(f"{cabin_safe}/{cabin_safe}_daily_max_summary.csv", dataframe_to_csv_bytes(table))

            excel_one = dataframe_to_excel_bytes({"Daily Max": table})
            zf.writestr(f"{cabin_safe}/{cabin_safe}_daily_max_summary.xlsx", excel_one)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="TOU Solar Daily Max Demand", layout="wide")

st.title("TOU Solar / Cabin Daily Maximum Demand Checker")
st.caption(
    "Upload one or more CSV files, calculate daily maximum kW, compare against contract kW, "
    "and export individual or combined reports."
)

uploaded_files = st.file_uploader(
    "Upload CSV file(s)",
    type=["csv"],
    accept_multiple_files=True,
    help="You can upload one file or many cabin CSV files at the same time.",
)

if not uploaded_files:
    st.info("Upload one or more CSV files to begin.")
    st.stop()

records = build_file_records(uploaded_files)
label_to_record = {record["label"]: record for record in records}

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose analysis mode",
    options=["Review one selected file", "Batch process selected files"],
)

# -----------------------------
# Single selected file mode
# -----------------------------
if mode == "Review one selected file":
    selected_label = st.sidebar.selectbox("Choose uploaded file", options=list(label_to_record.keys()))
    record = label_to_record[selected_label]

    try:
        raw_df = read_csv_flexible_from_bytes(record["bytes"])
    except Exception as exc:
        st.error(f"Could not read this CSV file: {exc}")
        st.stop()

    if raw_df.empty:
        st.error("The selected CSV is empty.")
        st.stop()

    columns = list(raw_df.columns)
    default_datetime_col = guess_datetime_col(columns)
    default_value_col = guess_value_col(columns)

    with st.expander("Preview selected file", expanded=False):
        st.write(f"**Filename:** `{record['filename']}`")
        st.write(f"**Detected cabin:** `{record['cabin']}`")
        st.dataframe(raw_df.head(20), use_container_width=True)

    st.sidebar.header("Analysis settings")
    cabin_name = st.sidebar.text_input("Cabin name", value=record["cabin"])
    contract_kw = st.sidebar.number_input("Contract kW", min_value=0.0, value=400.0, step=10.0)

    datetime_col = st.sidebar.selectbox(
        "Date/time column",
        options=columns,
        index=columns.index(default_datetime_col) if default_datetime_col in columns else 0,
    )
    value_col = st.sidebar.selectbox(
        "Power value column",
        options=columns,
        index=columns.index(default_value_col) if default_value_col in columns else 0,
    )
    unit_mode = st.sidebar.radio(
        "Input unit",
        options=["Already kW", "W → kW"],
        index=1 if guess_unit_mode(value_col) == "W → kW" else 0,
        help="Use 'W → kW' when the source column is Watt Total Avg. Use 'Already kW' when values already contain kW.",
    )

    result = process_daily_max(raw_df, datetime_col, value_col, unit_mode)

    if result.empty:
        st.error("No valid timestamp/kW rows found. Check the selected date/time and power columns.")
        st.stop()

    min_date = result["peak_timestamp"].dt.date.min()
    max_date = result["peak_timestamp"].dt.date.max()

    selected_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    filtered = filter_by_date(result, start_date, end_date)

    if filtered.empty:
        st.warning("No data found inside the selected date range.")
        st.stop()

    table = build_summary_table(filtered, contract_kw)

    peak_row = filtered.loc[filtered["daily_max_kW"].idxmax()]
    peak_kw = float(peak_row["daily_max_kW"])
    peak_time = peak_row["peak_timestamp"]
    exceed_days, max_excess = calculate_over_contract(filtered, contract_kw)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Maximum kW", f"{peak_kw:,.2f}")
    col2.metric("Peak time", peak_time.strftime(LABEL_FMT))
    col3.metric("Days over contract", exceed_days)
    col4.metric("Highest excess kW", f"{max_excess:,.2f}")

    if exceed_days > 0:
        st.warning(f"{cabin_name} exceeded the contract on {exceed_days} day(s). Highest excess: {max_excess:.2f} kW.")
    else:
        st.success(f"{cabin_name} stayed within the {contract_kw:g} kW contract for the selected period.")

    st.subheader("Daily Maximum Demand Chart")
    with _PLOT_LOCK:
        fig = make_chart(filtered, cabin_name, contract_kw, start_date, end_date)
        st.pyplot(fig, clear_figure=False)
        chart_png = fig_to_png_bytes(fig)
        plt.close(fig)

    st.subheader("Daily Maximum Demand Table")
    st.dataframe(
        table.style.apply(
            lambda row: ["background-color: #fff2cc" if row["Daily Max kW"] == table["Daily Max kW"].max() else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
    )

    table_png = make_table_png(table, title=f"Daily Max Table – {cabin_name}")
    csv_bytes = dataframe_to_csv_bytes(table)
    excel_bytes = dataframe_to_excel_bytes({"Daily Max": table})

    st.subheader("Export selected file")
    cabin_safe = safe_filename(cabin_name)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "Download chart PNG",
        data=chart_png,
        file_name=f"{cabin_safe}_daily_max_chart.png",
        mime="image/png",
        use_container_width=True,
    )
    d2.download_button(
        "Download table PNG",
        data=table_png,
        file_name=f"{cabin_safe}_daily_max_table.png",
        mime="image/png",
        use_container_width=True,
    )
    d3.download_button(
        "Download CSV summary",
        data=csv_bytes,
        file_name=f"{cabin_safe}_daily_max_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d4.download_button(
        "Download Excel summary",
        data=excel_bytes,
        file_name=f"{cabin_safe}_daily_max_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("Data quality checks"):
        parsed_ts = parse_datetime_series(raw_df[datetime_col])
        parsed_kw = convert_to_kw(raw_df[value_col], unit_mode)
        st.write(
            {
                "Total rows": int(len(raw_df)),
                "Valid timestamp rows": int(parsed_ts.notna().sum()),
                "Valid kW rows": int(parsed_kw.notna().sum()),
                "Rows used after cleaning": int(len(result)),
                "Source date min": str(min_date),
                "Source date max": str(max_date),
            }
        )

# -----------------------------
# Batch mode
# -----------------------------
else:
    st.subheader("Batch process uploaded files")

    selected_labels = st.multiselect(
        "Choose file(s) to process",
        options=list(label_to_record.keys()),
        default=list(label_to_record.keys()),
    )

    if not selected_labels:
        st.warning("Choose at least one file to process.")
        st.stop()

    selected_records = [label_to_record[label] for label in selected_labels]

    default_contract_kw = st.number_input(
        "Default contract kW for new/blank rows",
        min_value=0.0,
        value=400.0,
        step=10.0,
        help="Edit individual contract values in the table below.",
    )

    config_df = pd.DataFrame(
        {
            "Cabin": [record["cabin"] for record in selected_records],
            "Filename": [record["filename"] for record in selected_records],
            "Contract kW": [default_contract_kw for _ in selected_records],
        }
    )

    edited_config = st.data_editor(
        config_df,
        hide_index=True,
        use_container_width=True,
        disabled=["Cabin", "Filename"],
        column_config={
            "Contract kW": st.column_config.NumberColumn(
                "Contract kW",
                min_value=0.0,
                step=10.0,
                format="%.2f",
            )
        },
    )

    # First pass: detect available date range for selected files.
    precheck = []
    all_min_dates = []
    all_max_dates = []

    for record, (_, cfg_row) in zip(selected_records, edited_config.iterrows()):
        contract_kw = float(cfg_row["Contract kW"])
        analysis = analyze_record_auto(record, contract_kw)
        precheck.append(analysis)
        if analysis["ok"]:
            all_min_dates.append(analysis["result"]["peak_timestamp"].dt.date.min())
            all_max_dates.append(analysis["result"]["peak_timestamp"].dt.date.max())

    if not all_min_dates:
        st.error("None of the selected files could be processed. Check file formats and columns.")
        error_rows = [
            {
                "Cabin": a["record"]["cabin"],
                "Filename": a["record"]["filename"],
                "Error": a["error"],
            }
            for a in precheck
        ]
        st.dataframe(pd.DataFrame(error_rows), use_container_width=True)
        st.stop()

    date_mode = st.radio(
        "Date range option",
        options=["Use full date range from each file", "Use one date range for all selected files"],
        horizontal=True,
    )

    if date_mode == "Use one date range for all selected files":
        overall_min = min(all_min_dates)
        overall_max = max(all_max_dates)
        selected_range = st.date_input(
            "Common date range",
            value=(overall_min, overall_max),
            min_value=overall_min,
            max_value=overall_max,
        )

        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:
            start_date, end_date = overall_min, overall_max

        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()
    else:
        start_date, end_date = None, None

    if st.button("Run batch analysis", type="primary", use_container_width=True):
        successful = []
        failed = []
        summary_rows = []
        daily_tables = []

        for record, (_, cfg_row) in zip(selected_records, edited_config.iterrows()):
            contract_kw = float(cfg_row["Contract kW"])

            if date_mode == "Use one date range for all selected files":
                analysis = analyze_record_auto(record, contract_kw, start_date, end_date)
                row_start, row_end = start_date, end_date
            else:
                analysis = analyze_record_auto(record, contract_kw)
                if analysis["ok"]:
                    row_start = analysis["filtered"]["peak_timestamp"].dt.date.min()
                    row_end = analysis["filtered"]["peak_timestamp"].dt.date.max()
                else:
                    row_start, row_end = "", ""

            if analysis["ok"]:
                successful.append(analysis)
                summary_rows.append(summarize_analysis(analysis, row_start, row_end))

                daily = analysis["table"].copy()
                daily.insert(0, "Filename", record["filename"])
                daily.insert(0, "Cabin", record["cabin"])
                daily_tables.append(daily)
            else:
                failed.append(
                    {
                        "Cabin": record["cabin"],
                        "Filename": record["filename"],
                        "Error": analysis["error"],
                        "Detected Date Column": analysis["datetime_col"],
                        "Detected Value Column": analysis["value_col"],
                        "Detected Unit": analysis["unit_mode"],
                    }
                )

        if successful:
            combined_summary = pd.DataFrame(summary_rows)
            combined_daily = pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()

            ok_count = len(successful)
            fail_count = len(failed)
            over_count = int((combined_summary["Days Over Contract"] > 0).sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Processed files", ok_count)
            c2.metric("Failed files", fail_count)
            c3.metric("Cabins over contract", over_count)
            c4.metric("Highest kW found", f"{combined_summary['Maximum kW'].max():,.2f}")

            st.subheader("Combined batch summary")
            st.dataframe(
                combined_summary.style.apply(
                    lambda row: ["background-color: #fff2cc" if row["Days Over Contract"] > 0 else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )

            st.subheader("Combined daily max table")
            st.dataframe(combined_daily, use_container_width=True)

            bundle_start = start_date if start_date is not None else "full"
            bundle_end = end_date if end_date is not None else "range"
            zip_bytes = build_zip_bundle(successful, combined_summary, combined_daily, bundle_start, bundle_end)

            e1, e2, e3 = st.columns(3)
            e1.download_button(
                "Download combined summary CSV",
                data=dataframe_to_csv_bytes(combined_summary),
                file_name="combined_daily_max_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            e2.download_button(
                "Download combined Excel",
                data=dataframe_to_excel_bytes(
                    {
                        "Combined Summary": combined_summary,
                        "Combined Daily Max": combined_daily,
                    }
                ),
                file_name="combined_daily_max_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            e3.download_button(
                "Download ZIP report bundle",
                data=zip_bytes,
                file_name="daily_max_report_bundle.zip",
                mime="application/zip",
                use_container_width=True,
            )

            with st.expander("Show individual charts"):
                for analysis in successful:
                    record = analysis["record"]
                    st.markdown(f"### {record['cabin']} — `{record['filename']}`")
                    with _PLOT_LOCK:
                        chart_start = start_date if start_date is not None else analysis["filtered"]["peak_timestamp"].dt.date.min()
                        chart_end = end_date if end_date is not None else analysis["filtered"]["peak_timestamp"].dt.date.max()
                        fig = make_chart(
                            analysis["filtered"],
                            record["cabin"],
                            analysis["contract_kw"],
                            chart_start,
                            chart_end,
                        )
                        st.pyplot(fig, clear_figure=False)
                        plt.close(fig)

        if failed:
            st.subheader("Files that need attention")
            st.warning("These files were skipped. Do not ignore this table — fix the columns or review the file manually.")
            st.dataframe(pd.DataFrame(failed), use_container_width=True)

    with st.expander("Detected columns preview"):
        detected_rows = []
        for analysis in precheck:
            detected_rows.append(
                {
                    "Cabin": analysis["record"]["cabin"],
                    "Filename": analysis["record"]["filename"],
                    "Ready": analysis["ok"],
                    "Detected Date Column": analysis["datetime_col"],
                    "Detected Value Column": analysis["value_col"],
                    "Detected Unit": analysis["unit_mode"],
                    "Detected Date Min": (
                        str(analysis["result"]["peak_timestamp"].dt.date.min())
                        if analysis["ok"]
                        else ""
                    ),
                    "Detected Date Max": (
                        str(analysis["result"]["peak_timestamp"].dt.date.max())
                        if analysis["ok"]
                        else ""
                    ),
                    "Issue": analysis["error"],
                }
            )
        st.dataframe(pd.DataFrame(detected_rows), use_container_width=True)
