"""项目路径与全局配置。

所有脚本通过本模块定位资源，因此可以在任意目录下运行，
不再依赖「当前工作目录」这种脆弱约定。

用法：
    from config import DATA_DIR, MODEL_DIR, FIGURE_DIR, AGGREGATED_CSV
"""

from pathlib import Path

# 项目根目录（本文件所在 src/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"
DISPATCH_FIGURE_DIR = FIGURE_DIR / "dispatch"

# 原始刷卡数据（1.2 GB，未入库）。重跑 ETL 时放在此目录。
RAW_DIR = DATA_DIR / "raw"

AGGREGATED_CSV = DATA_DIR / "aggregated_data_full.csv.gz"
WEATHER_CSV = DATA_DIR / "weather.csv"
HOLIDAY_CSV = DATA_DIR / "holidays.csv"

# 模型输入特征（顺序与训练时保持一致）
FEATURES = ["hour", "LINECODE", "temp_celsius", "tp", "is_holiday", "weekday", "month"]

# 运力测算基准：8 米大巴荷载 54 人，实测早高峰单车约 30 人（荷载 56%），
# 取 35 人/车作为基准，约荷载 65%，为高峰留出余量。
PASSENGERS_PER_BUS = 35

RANDOM_STATE = 42


def ensure_dirs():
    """创建输出目录，供各脚本写模型与图表前调用。"""
    for d in (MODEL_DIR, FIGURE_DIR, DISPATCH_FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, hint: str = "") -> Path:
    """检查输入文件是否存在，缺失时给出可操作的提示而不是堆栈报错。"""
    if not path.exists():
        raise SystemExit(
            f"[缺少输入文件] {path}\n"
            f"{hint}\n"
            "请先执行 README 中的「复现方式」准备数据。"
        )
    return path
