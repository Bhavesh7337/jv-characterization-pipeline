from __future__ import annotations

import base64
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import PipelineConfig
from .types import AnalysisResult, Curve

METRIC_FIELDS = (
    "voc_v",
    "jsc_ma_cm2",
    "ff",
    "pce_percent",
    "pmax_mw_cm2",
    "vmp_v",
    "jmp_ma_cm2",
    "rs_ohm_cm2",
    "rsh_ohm_cm2",
    "s_kink_score",
    "hysteresis_index",
)


def result_row(result: AnalysisResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": str(result.source),
        "device_id": result.device_id,
        "scan": result.scan,
        "points": result.points,
        "sign_factor": result.sign_factor,
    }
    for field in METRIC_FIELDS:
        row[field] = getattr(result.metrics, field) if result.metrics is not None else math.nan
    row.update(
        {
            "rf_prediction": result.rf_prediction,
            "rf_anomaly_probability": result.rf_anomaly_probability,
            "anomalies": "; ".join(result.anomalies),
            "failed_rules": "; ".join(result.failed_rules),
            "warnings": "; ".join(result.warnings),
            "status": result.status,
            "error": result.error,
        }
    )
    return row


def result_record(result: AnalysisResult) -> dict[str, Any]:
    record = result_row(result)
    record["anomalies"] = list(result.anomalies)
    record["failed_rules"] = list(result.failed_rules)
    record["warnings"] = list(result.warnings)
    return record


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if math.isinf(value):
            return "∞"
        return f"{value:.{digits}f}"
    return str(value)


def _plot(curve: Curve, result: AnalysisResult) -> str:
    if not len(curve.voltage_v):
        return ""
    fig, axis = plt.subplots(figsize=(5.4, 3.5), dpi=130)
    axis.plot(curve.voltage_v, curve.current_density_ma_cm2, color="#167d9a", lw=2)
    axis.axhline(0, color="#718096", lw=0.7)
    axis.axvline(0, color="#718096", lw=0.7)
    if result.metrics is not None:
        axis.scatter(
            [result.metrics.vmp_v],
            [result.metrics.jmp_ma_cm2],
            color="#d1495b",
            s=30,
            label="MPP",
            zorder=3,
        )
        axis.legend(frameon=False)
    axis.set_xlabel("Voltage (V)")
    axis.set_ylabel("Generated current density (mA/cm²)")
    axis.set_title(f"{result.device_id} — {result.scan}")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_reports(
    output_dir: Path,
    curves: list[Curve],
    results: list[AnalysisResult],
    config: PipelineConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result_row(result) for result in results]
    pd.DataFrame(rows).to_csv(output_dir / "characterization_results.csv", index=False)
    payload = _json_safe([result_record(result) for result in results])
    (output_dir / "characterization_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_html(output_dir / "pass_fail_report.html", curves, results, config)


def _write_html(
    path: Path,
    curves: list[Curve],
    results: list[AnalysisResult],
    config: PipelineConfig,
) -> None:
    counts = {status: sum(result.status == status for result in results) for status in ("PASS", "FAIL", "ERROR")}
    rows: list[str] = []
    details: list[str] = []
    for curve, result in zip(curves, results):
        metrics = result.metrics
        values = {field: getattr(metrics, field) if metrics is not None else math.nan for field in METRIC_FIELDS}
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.device_id)}</td>"
            f"<td>{html.escape(result.scan)}</td>"
            f"<td>{_format(values['voc_v'])}</td>"
            f"<td>{_format(values['jsc_ma_cm2'])}</td>"
            f"<td>{_format(values['ff'])}</td>"
            f"<td>{_format(values['pce_percent'])}</td>"
            f"<td>{_format(values['rs_ohm_cm2'])}</td>"
            f"<td>{_format(values['rsh_ohm_cm2'])}</td>"
            f"<td>{html.escape(', '.join(result.anomalies) or '—')}</td>"
            f"<td class='{result.status.lower()}'>{result.status}</td>"
            "</tr>"
        )
        encoded = _plot(curve, result)
        image = f'<img src="data:image/png;base64,{encoded}" alt="J-V curve">' if encoded else ""
        assessment = result.error or "; ".join(result.failed_rules) or "All configured rules passed."
        details.append(
            "<section>"
            f"<h3>{html.escape(result.device_id)} — {html.escape(result.scan)}</h3>"
            f"{image}<p><strong>Assessment:</strong> {html.escape(assessment)}</p>"
            f"<p class='source'>{html.escape(str(result.source))}</p>"
            "</section>"
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>J-V characterization report</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;color:#172b36;background:#f4f8fa}}main{{max-width:1180px;margin:auto;padding:32px}}
h1{{margin-bottom:4px}}.muted,.source{{color:#657987;font-size:12px}}.summary{{display:flex;gap:24px;margin:24px 0}}.summary div{{font-size:17px}}.summary b{{display:block;font-size:28px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:9px;border:1px solid #d6e2e8;text-align:right}}th{{background:#0b1f2a;color:white}}td:first-child,td:nth-child(2),td:nth-child(9){{text-align:left}}.pass{{color:#268c6c;font-weight:700}}.fail,.error{{color:#d1495b;font-weight:700}}
.details{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-top:16px}}section{{background:white;padding:18px}}section img{{max-width:100%}}details{{margin-top:24px}}pre{{white-space:pre-wrap;background:#0b1f2a;color:#d6e2e8;padding:16px}}
</style></head><body><main>
<h1>Automated J-V characterization report</h1><div class="muted">Generated {datetime.now().astimezone().isoformat(timespec='seconds')}</div>
<div class="summary"><div><b>{len(results)}</b>curves</div><div><b class="pass">{counts['PASS']}</b>passed</div><div><b class="fail">{counts['FAIL']}</b>failed</div><div><b class="error">{counts['ERROR']}</b>errors</div></div>
<h2>Summary</h2><table><thead><tr><th>Device</th><th>Scan</th><th>Voc (V)</th><th>Jsc (mA/cm²)</th><th>FF</th><th>PCE (%)</th><th>Rs (Ω·cm²)</th><th>Rsh (Ω·cm²)</th><th>Anomalies</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Curve review</h2><div class="details">{''.join(details)}</div>
<details><summary>Configuration</summary><pre>{html.escape(json.dumps(config.to_dict(), indent=2))}</pre></details>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")
