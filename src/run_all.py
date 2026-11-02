"""一键运行完整分析与建模流程。

用法：
    python src/run_all.py              # 训练 + 出图（使用仓库内聚合数据）
    python src/run_all.py --with-etl   # 额外从原始流水重跑 ETL（需自备 1.2 GB 原始数据）

前置：仓库内 data/aggregated_data_full.csv.gz 已就位（随仓库提供）。
产出：models/ 下的模型文件、figures/ 下的分析图表。
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

# (脚本, 说明)
PIPELINE = [
    ("02_model_training.py", "训练基线模型（线性回归 + 随机森林）"),
    ("02e_stratified_model.py", "训练三层分层模型（工作日 / 周末 / 节假日）"),
    ("03_visualization.py", "生成分析与调度图表"),
    ("04_dispatch_optimization.py", "生成调度优化图表与发车间隔"),
]

ETL_STEP = ("01_data_preprocessing.py", "从原始流水重跑 ETL 聚合（可选）")


def run_step(script: str, desc: str, index: str) -> None:
    print(f"\n{'=' * 64}\n[{index}] {desc}\n{'=' * 64}", flush=True)
    result = subprocess.run([sys.executable, str(SRC_DIR / script)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"\n[失败] {script} 退出码 {result.returncode}，流程中止。")


def main() -> None:
    steps = list(PIPELINE)
    if "--with-etl" in sys.argv:
        steps.insert(0, ETL_STEP)

    print(f"Python 解释器：{sys.executable}")
    print(f"项目根目录：  {ROOT}")

    for i, (script, desc) in enumerate(steps, 1):
        run_step(script, desc, f"{i}/{len(steps)}")

    print(f"\n{'=' * 64}")
    print("全部完成")
    print(f"  模型：{ROOT / 'models'}")
    print(f"  图表：{ROOT / 'figures'}")
    print("\n如需图形化预测系统（需安装 PyQt5）：")
    print("  python src/05_prediction_gui.py")


if __name__ == "__main__":
    main()
