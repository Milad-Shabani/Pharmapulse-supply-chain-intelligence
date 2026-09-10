"""
Builds the planner-facing Excel workbook
(`PharmaPulse_Planning_Report.xlsx`) - the artifact a demand-planning,
supply-chain, or finance analyst would actually receive at the end of
a forecasting cycle, with native (editable) Excel charts rather than
pasted images, formatted headers, and frozen panes for usability.

The same computed tables that populate this workbook also feed
`reporting.html_dashboard`, so the two outputs are guaranteed to be
showing the same numbers (a single source of truth, as requested).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


HEADER_FMT = dict(bold=True, bg_color="#C0392B", font_color="white", border=1, text_wrap=True, valign="vcenter")


def _write_df(writer, df: pd.DataFrame, sheet_name: str, freeze_header: bool = True, col_widths: dict | None = None):
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1, header=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    header_format = workbook.add_format(HEADER_FMT)
    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, col_name, header_format)
        width = (col_widths or {}).get(col_name, max(12, min(32, len(str(col_name)) + 4)))
        worksheet.set_column(col_idx, col_idx, width)
    if freeze_header:
        worksheet.freeze_panes(1, 0)
    return worksheet


def build_excel_report(
    output_path: Path,
    exec_summary: dict,
    revenue_by_category: pd.DataFrame,
    fill_rate_by_month: pd.DataFrame,
    forecast_by_category_week: pd.DataFrame,
    inventory_plan: pd.DataFrame,
    abc_xyz: pd.DataFrame,
    expiry_risk: pd.DataFrame,
    model_performance: pd.DataFrame,
    ccc_summary: dict,
    working_capital_timeseries: pd.DataFrame,
    supplier_performance: pd.DataFrame,
    material_coverage: pd.DataFrame,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # ---------------- Executive Summary ----------------
        ws = workbook.add_worksheet("Executive Summary")
        writer.sheets["Executive Summary"] = ws
        title_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#7B241C"})
        label_fmt = workbook.add_format({"bold": True, "font_size": 11})
        value_fmt = workbook.add_format({"font_size": 20, "bold": True, "font_color": "#C0392B"})
        sub_fmt = workbook.add_format({"italic": True, "font_color": "#9A6B58"})

        ws.write(0, 0, "PharmaPulse — Demand Planning & Inventory Intelligence", title_fmt)
        ws.write(1, 0, "Aurora Pharmaceuticals · Network-wide report (synthetic demo data)", sub_fmt)

        kpi_row = 3
        for i, (label, value) in enumerate(exec_summary.items()):
            col = i * 2
            ws.write(kpi_row, col, label, label_fmt)
            ws.write(kpi_row + 1, col, value, value_fmt)
        ws.set_column(0, 20, 20)

        rev_start_row = kpi_row + 4
        ws.write(rev_start_row, 0, "Revenue by Therapeutic Category", label_fmt)
        for i, r in enumerate(revenue_by_category.itertuples(index=False), start=rev_start_row + 1):
            ws.write(i, 0, r.category)
            ws.write(i, 1, float(r.total_revenue))
        chart1 = workbook.add_chart({"type": "bar"})
        n = len(revenue_by_category)
        chart1.add_series({
            "name": "Revenue",
            "categories": ["Executive Summary", rev_start_row + 1, 0, rev_start_row + n, 0],
            "values": ["Executive Summary", rev_start_row + 1, 1, rev_start_row + n, 1],
            "fill": {"color": "#C0392B"},
        })
        chart1.set_title({"name": "Revenue by Category"})
        chart1.set_legend({"none": True})
        ws.insert_chart(rev_start_row, 3, chart1, {"x_scale": 1.3, "y_scale": 1.3})

        fr_start_row = rev_start_row + n + 3
        ws.write(fr_start_row, 0, "Network Fill Rate by Month", label_fmt)
        for i, r in enumerate(fill_rate_by_month.itertuples(index=False), start=fr_start_row + 1):
            ws.write(i, 0, str(r.month))
            ws.write(i, 1, float(r.fill_rate))
        m = len(fill_rate_by_month)
        chart2 = workbook.add_chart({"type": "line"})
        chart2.add_series({
            "name": "Fill rate",
            "categories": ["Executive Summary", fr_start_row + 1, 0, fr_start_row + m, 0],
            "values": ["Executive Summary", fr_start_row + 1, 1, fr_start_row + m, 1],
            "line": {"color": "#E67E22", "width": 2.5},
        })
        chart2.set_title({"name": "Network Fill Rate Trend"})
        chart2.set_y_axis({"num_format": "0%", "min": 0.5, "max": 1.0})
        ws.insert_chart(fr_start_row, 3, chart2, {"x_scale": 1.3, "y_scale": 1.3})

        # CCC mini-summary block
        ccc_start_row = fr_start_row + m + 3
        ws.write(ccc_start_row, 0, "Cash Conversion Cycle (60-day trailing)", label_fmt)
        ccc_labels = ["DIO (days)", "DSO (days)", "DPO (days)", "CCC (days)"]
        ccc_values = [ccc_summary["DIO_days"], ccc_summary["DSO_days"], ccc_summary["DPO_days"], ccc_summary["CCC_days"]]
        for i, (lbl, val) in enumerate(zip(ccc_labels, ccc_values)):
            ws.write(ccc_start_row + 1 + i, 0, lbl)
            ws.write(ccc_start_row + 1 + i, 1, float(val))
        ws.write(ccc_start_row + 6, 0, f"As of {ccc_summary['as_of_date']}", sub_fmt)

        _write_df(writer, forecast_by_category_week, "Forecast by Category")
        _write_df(writer, inventory_plan, "Inventory Plan", col_widths={"generic_name": 26})
        _write_df(writer, abc_xyz, "ABC-XYZ Segmentation")
        _write_df(writer, expiry_risk, "Expiry Risk", col_widths={"generic_name": 26})

        # ---------------- Working Capital (CCC) ----------------
        wc_ws = _write_df(
            writer,
            working_capital_timeseries[["date", "DIO", "DSO", "DPO", "CCC"]].dropna(),
            "Working Capital (CCC)",
        )
        wc_n = working_capital_timeseries["CCC"].notna().sum()
        chart3 = workbook.add_chart({"type": "line"})
        for col_idx, (col_name, color) in enumerate(
            zip(["DIO", "DSO", "DPO", "CCC"], ["#E67E22", "#C0392B", "#27AE60", "#7B241C"]), start=1
        ):
            chart3.add_series({
                "name": col_name,
                "categories": ["Working Capital (CCC)", 1, 0, wc_n, 0],
                "values": ["Working Capital (CCC)", 1, col_idx, wc_n, col_idx],
                "line": {"color": color, "width": 2},
            })
        chart3.set_title({"name": "CCC & Components Over Time"})
        chart3.set_y_axis({"name": "Days"})
        wc_ws.insert_chart(1, 6, chart3, {"x_scale": 1.8, "y_scale": 1.6})

        # ---------------- Raw Materials & Suppliers ----------------
        sup_ws = _write_df(writer, supplier_performance, "Suppliers", col_widths={"supplier_name": 26})
        chart4 = workbook.add_chart({"type": "bar"})
        sn = len(supplier_performance)
        on_time_col = supplier_performance.columns.get_loc("on_time_rate")
        name_col = supplier_performance.columns.get_loc("supplier_name")
        chart4.add_series({
            "name": "On-time rate",
            "categories": ["Suppliers", 1, name_col, sn, name_col],
            "values": ["Suppliers", 1, on_time_col, sn, on_time_col],
            "fill": {"color": "#E67E22"},
        })
        chart4.set_title({"name": "Supplier On-Time Delivery Rate"})
        chart4.set_x_axis({"num_format": "0%"})
        chart4.set_legend({"none": True})
        sup_ws.insert_chart(1, len(supplier_performance.columns) + 2, chart4, {"x_scale": 1.3, "y_scale": 1.6})

        _write_df(writer, material_coverage, "Raw Material Coverage", col_widths={"material_name": 28})

        _write_df(writer, model_performance.reset_index(), "Model Performance")

    return output_path
