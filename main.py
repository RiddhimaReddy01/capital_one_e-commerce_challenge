from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

# Simple logging to both console and file
class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(log_file, "w", encoding="utf-8")

    def log(self, message: str):
        """Print to console AND save to file"""
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        """Close log file when done"""
        self.file.close()

from src.dashboard import build_dashboard
from src.eda_artifacts import build_eda_artifacts
from src.io_utils import ensure_dir, write_table
from src.metadata_validation import validate_metadata_registry
from src.metrics import build_metrics
from src.pipeline import build_final_dataset_line, build_order_level
from src.quality import run_quality_checks
from src.config import Config
from src.reporting import write_executive_summary, write_metadata_fields, write_processing_log
from src.transform import build_clean_layer
from src.validate import (
    validate_buyers_raw,
    validate_sales_raw,
    validate_products_raw,
    validate_cross_dataset,
)


def load_raw_data(settings: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales_file = settings.raw_data_path / "sales.csv"
    buyers_file = settings.raw_data_path / "buyer.csv"

    if not sales_file.exists():
        raise FileNotFoundError(f"Missing raw sales.csv at {sales_file}")
    if not buyers_file.exists():
        raise FileNotFoundError(f"Missing raw buyer.csv at {buyers_file}")

    sales_raw = pd.read_csv(sales_file)
    buyers_raw = pd.read_csv(buyers_file)

    # Products are loaded via glob in clean_products, but we need raw files for validate_products_raw
    vendor_dir = settings.raw_data_path / "Vendor Datasets"
    if not vendor_dir.exists():
        raise FileNotFoundError(f"Missing raw Vendor Datasets directory at {vendor_dir}")

    import glob as glob_module

    files = glob_module.glob(str(vendor_dir / "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {vendor_dir}")

    products_raw = pd.concat(
        [pd.read_csv(p) for p in sorted(files)], ignore_index=True
    )

    return sales_raw, buyers_raw, products_raw


def main() -> None:
    # Create logger that saves to file
    log_path = Path("outputs/pipeline.log")
    logger = Logger(log_path)

    logger.log("="*70)
    logger.log("CAPITAL ONE CHALLENGE - DATA PIPELINE")
    logger.log("="*70)
    logger.log(f"Started at: {datetime.now(timezone.utc).isoformat()}")

    settings = Config.from_env()
    logger.log(f"\n[main] Config loaded - checking directories...")
    ensure_dir(settings.processed_dir)
    ensure_dir(settings.metrics_dir)
    ensure_dir(settings.reports_dir)
    ensure_dir(settings.dashboard_dir)
    ensure_dir(settings.processed_dir / "eda")
    logger.log(f"[main] All output directories ready")

    logger.log("\n[1/11] Loading raw data...")
    sales_raw, buyers_raw, products_raw = load_raw_data(settings)
    logger.log(f"sales_raw: {len(sales_raw):,} rows, buyers_raw: {len(buyers_raw):,} rows, products_raw: {len(products_raw):,} rows")

    logger.log("\n[2/11] Validating raw datasets...")
    raw_checks = []
    raw_checks.extend(validate_buyers_raw(buyers_raw))
    raw_checks.extend(validate_sales_raw(sales_raw))
    raw_checks.extend(validate_products_raw(products_raw))
    raw_passed = all(c.passed for c in raw_checks if c.severity in {"HIGH", "CRITICAL"})
    logger.log(f"Raw validation: {'PASSED' if raw_passed else 'WARNINGS'}")
    for c in raw_checks:
        if not c.passed and c.severity in {"HIGH", "CRITICAL"}:
            logger.log(f"{c.rule_id}: {c.description} (value={c.value})")

    logger.log("\n[3/11] Validating cross-dataset relationships...")
    cross_checks = validate_cross_dataset(sales_raw, buyers_raw, products_raw, settings)
    cross_passed = all(c.passed for c in cross_checks if c.severity in {"HIGH", "CRITICAL"})
    logger.log(f"Cross-dataset validation: {'PASSED' if cross_passed else 'WARNINGS'}")
    for c in cross_checks:
        if not c.passed:
            logger.log(f"{c.rule_id}: {c.description} (coverage={c.value}%)")

    logger.log("\n[4/11] Cleaning datasets...")
    buyers, sales, products = build_clean_layer(settings)
    logger.log(f"[main] OK buyers_clean: {len(buyers):,} rows")
    logger.log(f"[main] OK sales_clean: {len(sales):,} rows")
    logger.log(f"[main] OK products_clean: {len(products):,} rows")

    logger.log("\n[5/11] Joining datasets...")
    final_line = build_final_dataset_line(buyers, sales, products, settings)
    logger.log(f"[main] OK final_dataset_line: {len(final_line):,} rows")

    logger.log("\n[6/11] Building order-level dataset...")
    order_level = build_order_level(final_line, settings)
    logger.log(f"[main] OK order_level: {len(order_level):,} orders")

    logger.log("\n[main] Writing tables to parquet and CSV...")
    write_table(final_line, settings.processed_dir / "final_line_level")
    write_table(order_level, settings.processed_dir / "final_order_level")

    logger.log("\n[7/11] Running quality checks...")
    quality = run_quality_checks(buyers, sales, products, final_line, order_level, settings)
    logger.log(f"[main] OK Quality checks: {'PASSED' if quality['passed'] else 'WARNINGS'}")
    logger.log(f"[main]   - Join coverage: {quality['summary']['coverage_pct']:.2f}%")
    logger.log(f"[main]   - Row retention: {quality['summary']['row_retention_pct']:.2f}%")

    logger.log("\n[8/11] Generating EDA artifacts...")
    eda_outputs = build_eda_artifacts(
        buyers, sales, products, settings.processed_dir, final_line, order_level
    )
    logger.log("[main] OK EDA artifacts generated")

    logger.log("\n[9/11] Calculating business metrics...")
    metrics = build_metrics(order_level, settings)
    logger.log(f"[main] OK {len(metrics)} metric tables generated:")
    for name in metrics:
        logger.log(f"[main]   - {name}: {len(metrics[name]):,} rows")

    logger.log("\n[10/11] Validating metadata registry...")
    metadata_report = validate_metadata_registry(settings, buyers, final_line, order_level)
    logger.log(f"Metadata validation: {'PASSED' if metadata_report['passed'] else 'WARNINGS'}")

    print("\n[11/11] Writing run manifest and generating dashboard...")
    run_manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings.as_dict(),
        "validation": {
            "raw_checks_passed": raw_passed,
            "cross_checks_passed": cross_passed,
            "raw_checks_count": len(raw_checks),
            "cross_checks_count": len(cross_checks),
        },
        "outputs": {
            "buyers_clean_rows": int(len(buyers)),
            "sales_clean_rows": int(len(sales)),
            "products_clean_rows": int(len(products)),
            "final_dataset_line_rows": int(len(final_line)),
            "order_level_rows": int(len(order_level)),
            "metric_tables": {name: int(len(df)) for name, df in metrics.items()},
            "eda_artifacts": {
                "buyers_summary_rows": int(eda_outputs["buyers"]["rows"]),
                "sales_summary_rows": int(eda_outputs["sales"]["rows"]),
                "products_summary_rows": int(eda_outputs["products"]["rows"]),
                "final_line_summary_rows": int(eda_outputs.get("final_line", {}).get("rows", 0)),
                "order_level_summary_rows": int(eda_outputs.get("order_level", {}).get("rows", 0)),
                "eda_dir": str(settings.processed_dir / "eda"),
                "drift_report": str(settings.processed_dir / "eda" / "eda_drift_report.json"),
            },
        },
        "quality_passed": bool(quality["passed"]),
        "metadata_validation_passed": bool(metadata_report["passed"]),
        "libraries": [
            "pandas",
            "numpy",
            "pyarrow",
            "nbformat",
            "matplotlib",
            "seaborn",
            "plotly",
        ],
    }
    write_processing_log(run_manifest, settings)

    registry_fields: dict = {"fields": []}
    registry_path = Path(settings.metadata_registry_path)
    if registry_path.exists():
        try:
            registry_fields = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry_fields = {"fields": [], "error": f"Invalid JSON in {registry_path}"}
    registry_fields["validation_passed"] = bool(metadata_report["passed"])
    write_metadata_fields(registry_fields, settings)

    summary_payload = {
        "pipeline_status": "COMPLETE",
        "quality_passed": bool(quality["passed"]),
        "metadata_validation_passed": bool(metadata_report["passed"]),
        "line_level_rows": int(len(final_line)),
        "order_level_rows": int(len(order_level)),
        "join_coverage_pct": float(quality["summary"]["coverage_pct"]),
        "dashboard": str(settings.dashboard_dir / "index.html"),
    }
    write_executive_summary(summary_payload, settings)

    # Generate dashboard
    logger.log("\n[11/11] Generating executive dashboard...")
    build_dashboard(settings.metrics_dir, settings.dashboard_dir, order_level, quality["summary"]["coverage_pct"])
    dashboard_file = settings.dashboard_dir / "index.html"
    logger.log(f"[main] OK Dashboard generated: {dashboard_file}")

    logger.log("\n" + "="*70)
    logger.log("PIPELINE COMPLETE")
    logger.log("="*70)
    logger.log(f"\nData Summary:")
    logger.log(f"  Buyers: {len(buyers):,}")
    logger.log(f"  Sales: {len(sales):,}")
    logger.log(f"  Products: {len(products):,}")
    logger.log(f"  Final line-level: {len(final_line):,} lines")
    logger.log(f"  Order-level: {len(order_level):,} orders")
    logger.log(f"\nQuality Metrics:")
    logger.log(f"  Join coverage: {quality['summary']['coverage_pct']:.2f}%")
    logger.log(f"  Row retention: {quality['summary']['row_retention_pct']:.2f}%")
    logger.log(f"  Status: {'PASSED' if quality['passed'] else 'WARNINGS'}")
    logger.log(f"\nOutputs:")
    logger.log(f"  Parquet data: {settings.processed_dir / 'final_line_level.parquet'}")
    logger.log(f"  CSV data: {settings.processed_dir / 'final_line_level.csv'} and {settings.processed_dir / 'final_order_level.csv'}")
    logger.log(f"  Reports: {settings.reports_dir}")
    logger.log(f"  Dashboard: {dashboard_file}")
    logger.log(f"  Log file: {log_path}")
    logger.log("="*70)
    logger.log(f"Finished at: {datetime.now(timezone.utc).isoformat()}")

    # Close log file
    logger.close()


if __name__ == "__main__":
    main()
