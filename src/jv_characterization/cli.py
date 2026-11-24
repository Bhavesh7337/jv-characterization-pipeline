from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config, write_default_config
from .model import train_model
from .pipeline import analyze


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jv-characterize",
        description="Extract photovoltaic J-V metrics and screen device anomalies.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    analyze_parser = commands.add_parser("analyze", help="analyze a file or directory")
    analyze_parser.add_argument("input", type=Path, help="J-V file or directory")
    analyze_parser.add_argument("-o", "--output", type=Path, default=Path("report"))
    analyze_parser.add_argument("-c", "--config", type=Path)
    analyze_parser.add_argument("--model", type=Path, help="trusted Random Forest model")

    train_parser = commands.add_parser("train", help="train a Random Forest detector")
    train_parser.add_argument("training_csv", type=Path)
    train_parser.add_argument("--model", type=Path, required=True, help="output model path")
    train_parser.add_argument("-c", "--config", type=Path)
    train_parser.add_argument("--label-column", default="anomaly_label")
    train_parser.add_argument("--trees", type=int, default=400)
    train_parser.add_argument("--random-state", type=int, default=42)

    config_parser = commands.add_parser("write-config", help="write the default configuration")
    config_parser.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-config":
            write_default_config(args.path)
            print(args.path.resolve())
            return 0

        config = load_config(args.config)
        if args.command == "train":
            if args.trees < 10:
                raise ValueError("--trees must be at least 10")
            summary = train_model(
                args.training_csv,
                args.model,
                config,
                label_column=args.label_column,
                n_estimators=args.trees,
                random_state=args.random_state,
            )
            print(
                json.dumps(
                    {
                        "model": str(args.model.resolve()),
                        "rows": summary.rows,
                        "classes": summary.classes,
                        "oob_score": round(summary.oob_score, 4),
                    },
                    indent=2,
                )
            )
            return 0

        results = analyze(args.input, args.output, config, args.model)
        counts = {
            status: sum(result.status == status for result in results)
            for status in ("PASS", "FAIL", "ERROR")
        }
        print(
            f"Processed {len(results)} curve(s): "
            f"{counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['ERROR']} ERROR"
        )
        print((args.output / "pass_fail_report.html").resolve())
        return 1 if counts["ERROR"] else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
