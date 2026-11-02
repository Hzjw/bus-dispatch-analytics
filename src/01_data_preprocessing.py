"""01 数据预处理：原始刷卡记录 → 聚合训练集

输入：
    data/raw/bus_card_records.csv   原始刷卡流水（约 1,596 万条，未入库）
    data/weather.csv                小时级气象数据
    data/holidays.csv               法定节假日标注

输出：
    data/aggregated_data.csv        「日期 × 小时 × 线路」聚合表

注意：原始流水约 1.2 GB，因体积与授权限制未随仓库分发。
若只想复现建模与可视化，可直接使用仓库内的 data/aggregated_data_full.csv.gz，
无需执行本脚本。
"""

import warnings

import pandas as pd

from config import DATA_DIR, HOLIDAY_CSV, RAW_DIR, WEATHER_CSV, require_file

warnings.filterwarnings("ignore")

# 原始刷卡数据支持中英文两种文件名，方便直接拷贝原始文件
BUS_RAW_CANDIDATES = [
    RAW_DIR / "bus_card_records.csv",
    RAW_DIR / "公交刷卡_2023-2024数据_去重后.csv",
]


def load_bus_raw() -> pd.DataFrame:
    for path in BUS_RAW_CANDIDATES:
        if path.exists():
            print(f"读取原始刷卡流水：{path}")
            return pd.read_csv(path)
    raise SystemExit(
        f"[缺少原始刷卡数据] 请在 {RAW_DIR} 下放入原始流水文件，任选其一：\n"
        + "\n".join(f"  - {p.name}" for p in BUS_RAW_CANDIDATES)
        + "\n\n原始数据约 1.2 GB，未随仓库分发。"
        "\n只想跑通建模与可视化的话，直接用 data/aggregated_data_full.csv.gz 即可，"
        "不必执行本脚本。"
    )


print("1. 读取数据...")
df_bus = load_bus_raw()
df_weather = pd.read_csv(require_file(WEATHER_CSV))
df_holiday = pd.read_csv(require_file(HOLIDAY_CSV))

print(f"   公交卡数据：{df_bus.shape}")
print(f"   天气数据：  {df_weather.shape}")
print(f"   节假日数据：{df_holiday.shape}")

print("\n2. 解析时间字段...")
df_bus["CONSUMEDATETIME"] = pd.to_datetime(df_bus["CONSUMEDATETIME"])
df_bus["hour"] = df_bus["CONSUMEDATETIME"].dt.hour
df_bus["date"] = df_bus["CONSUMEDATETIME"].dt.date.astype(str)

print("3. 按「日期 × 小时 × 线路」聚合...")
# ride_count：该时段刷卡乘次
# bus_count： 该时段实际运营的不同车辆数（BUSCODE 唯一计数），供调度分析使用
agg_spec = {"ride_count": ("CONSUMEDATETIME", "size")}
if "BUSCODE" in df_bus.columns:
    agg_spec["bus_count"] = ("BUSCODE", "nunique")
else:
    print("   [提示] 原始数据无 BUSCODE 列，跳过 bus_count 统计")

df_agg = df_bus.groupby(["date", "hour", "LINECODE"]).agg(**agg_spec).reset_index()
print(f"   聚合后：{df_agg.shape}")

print("4. 关联天气数据...")
df_weather["valid_time"] = pd.to_datetime(df_weather["valid_time"])
df_weather["date"] = df_weather["valid_time"].dt.date.astype(str)
df_weather["weather_hour"] = df_weather["valid_time"].dt.hour
df_agg = df_agg.merge(
    df_weather[["date", "weather_hour", "t2m", "tp"]],
    left_on=["date", "hour"],
    right_on=["date", "weather_hour"],
    how="left",
)

print("5. 关联节假日数据...")
df_holiday["date"] = df_holiday["日期"].astype(str)
df_agg = df_agg.merge(df_holiday[["date", "是否节假日"]], on="date", how="left")

print("6. 特征工程...")
df_agg["is_holiday"] = (df_agg["是否节假日"] == "是").astype(int)
df_agg["temp_celsius"] = df_agg["t2m"] - 273.15  # 开尔文 → 摄氏度
df_agg["date_dt"] = pd.to_datetime(df_agg["date"])
df_agg["weekday"] = df_agg["date_dt"].dt.dayofweek  # 0=周一
df_agg["month"] = df_agg["date_dt"].dt.month
df_agg = df_agg.drop(columns=["date_dt"]).dropna()

print(f"\n最终数据形状：{df_agg.shape}")
print("\n数据示例：")
print(
    df_agg[
        ["date", "hour", "LINECODE", "ride_count", "temp_celsius", "tp", "is_holiday", "weekday"]
    ].head(10)
)

out_path = DATA_DIR / "aggregated_data.csv"
df_agg.to_csv(out_path, index=False)
print(f"\n聚合数据已保存：{out_path}")
