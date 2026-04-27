from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os

# Schema: required columns per grain (shared across validate + tests)
BUYERS_REQUIRED_COLS = ("buyer_id", "customer_segment", "state", "region")
LINE_REQUIRED_COLS = ("order_id", "buyer_id", "sku_id", "order_datetime")
ORDER_REQUIRED_COLS = ("order_id", "buyer_id", "order_revenue", "order_profit", "quarter", "hour_company")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Config:
    raw_data_path: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    metadata_registry_path: Path = Path("metadata/derived_fields.json")

    # Business logic parameters
    company_timezone: str = "America/Chicago"
    referral_discount_rate: float = 0.10
    black_friday_discount_rate: float = 0.20
    company_shipping_cost: float = 4.99

    shipping_threshold_1: float = 50.0
    shipping_threshold_2: float = 100.0
    shipping_fee_low: float = 7.99
    shipping_fee_mid: float = 4.99
    shipping_fee_high: float = 0.0

    min_join_coverage_pct: float = 90.0
    buyers_coverage_filter: bool = True

    def __post_init__(self):
        # Validate configuration
        if self.shipping_threshold_1 >= self.shipping_threshold_2:
            raise ValueError(f"ERROR: shipping_threshold_1 ({self.shipping_threshold_1}) must be less than shipping_threshold_2 ({self.shipping_threshold_2})")
        if self.shipping_fee_low < self.shipping_fee_mid:
            print(f"[Config] WARNING: shipping_fee_low ({self.shipping_fee_low}) is less than shipping_fee_mid ({self.shipping_fee_mid})")
        if self.referral_discount_rate < 0 or self.referral_discount_rate > 1:
            raise ValueError(f"ERROR: referral_discount_rate must be between 0 and 1, got {self.referral_discount_rate}")
        if self.black_friday_discount_rate < 0 or self.black_friday_discount_rate > 1:
            raise ValueError(f"ERROR: black_friday_discount_rate must be between 0 and 1, got {self.black_friday_discount_rate}")
        if self.min_join_coverage_pct < 0 or self.min_join_coverage_pct > 100:
            raise ValueError(f"ERROR: min_join_coverage_pct must be between 0 and 100, got {self.min_join_coverage_pct}")

    @property
    def dashboard_dir(self) -> Path:
        return self.outputs_dir / "dashboard"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def metrics_dir(self) -> Path:
        return self.processed_dir / "metrics"

    @property
    def metadata_dir(self) -> Path:
        return self.metadata_registry_path.parent

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            raw_data_path=Path(os.getenv("CAPITAL_ONE_RAW_DATA_PATH", str(cls.raw_data_path))),
            processed_dir=Path(os.getenv("CAPITAL_ONE_PROCESSED_DIR", str(cls.processed_dir))),
            outputs_dir=Path(os.getenv("CAPITAL_ONE_OUTPUTS_DIR", str(cls.outputs_dir))),
            metadata_registry_path=Path(
                os.getenv("CAPITAL_ONE_METADATA_REGISTRY_PATH", str(cls.metadata_registry_path))
            ),
            company_timezone=os.getenv("CAPITAL_ONE_COMPANY_TIMEZONE", cls.company_timezone),
            referral_discount_rate=_env_float("CAPITAL_ONE_REFERRAL_DISCOUNT_RATE", cls.referral_discount_rate),
            black_friday_discount_rate=_env_float(
                "CAPITAL_ONE_BLACK_FRIDAY_DISCOUNT_RATE", cls.black_friday_discount_rate
            ),
            company_shipping_cost=_env_float("CAPITAL_ONE_COMPANY_SHIPPING_COST", cls.company_shipping_cost),
            shipping_threshold_1=_env_float("CAPITAL_ONE_SHIPPING_THRESHOLD_1", cls.shipping_threshold_1),
            shipping_threshold_2=_env_float("CAPITAL_ONE_SHIPPING_THRESHOLD_2", cls.shipping_threshold_2),
            shipping_fee_low=_env_float("CAPITAL_ONE_SHIPPING_FEE_LOW", cls.shipping_fee_low),
            shipping_fee_mid=_env_float("CAPITAL_ONE_SHIPPING_FEE_MID", cls.shipping_fee_mid),
            shipping_fee_high=_env_float("CAPITAL_ONE_SHIPPING_FEE_HIGH", cls.shipping_fee_high),
            min_join_coverage_pct=_env_float("CAPITAL_ONE_MIN_JOIN_COVERAGE_PCT", cls.min_join_coverage_pct),
            buyers_coverage_filter=_env_bool("CAPITAL_ONE_BUYERS_COVERAGE_FILTER", cls.buyers_coverage_filter),
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        # Convert Path to string
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)
        # Add derived paths
        data["dashboard_dir"] = str(self.dashboard_dir)
        data["reports_dir"] = str(self.reports_dir)
        data["metrics_dir"] = str(self.metrics_dir)
        data["metadata_dir"] = str(self.metadata_dir)
        data["metadata_registry_path"] = str(self.metadata_registry_path)
        return data
