# 数据说明

## 已入库的数据

| 文件 | 大小 | 说明 |
| :--- | ---: | :--- |
| `aggregated_data_full.csv.gz` | 2.1 MB | 聚合后训练集，**419,045 条 × 13 列**。gzip 压缩，解压后约 28 MB。pandas 可直接读取，无需手动解压 |
| `sample_aggregated.csv` | 199 KB | 上述文件前 3,000 行，仅用于快速查看字段格式，不足以训练 |
| `weather.csv` | 968 KB | 小时级气象数据（温度、降水量） |
| `holidays.csv` | 24 KB | 2023–2024 年法定节假日标注 |

### 主数据集字段

| 字段 | 含义 | 类型 |
| :--- | :--- | :--- |
| `date` | 日期 | string |
| `hour` | 小时（0–23） | int |
| `LINECODE` | 线路编号（共 54 条线路） | string |
| `ride_count` | 该「日期×小时×线路」的刷卡乘次 | int |
| `bus_count` | 该时段实际运营车辆数（BUSCODE 唯一计数） | int |
| `weather_hour` | 气象记录小时 | int |
| `t2m` | 气温（开尔文） | float |
| `temp_celsius` | 气温（摄氏度） | float |
| `tp` | 降水量（mm/h） | float |
| `是否节假日` / `is_holiday` | 是否法定节假日 | 0/1 |
| `weekday` | 星期（0=周一） | int |
| `month` | 月份（1–12） | int |

解压命令：

```bash
gunzip -k aggregated_data_full.csv.gz
```

## 未入库的数据

**原始刷卡记录**（`公交刷卡_2023-2024数据.csv`，约 1.2 GB，1,596 万条）因体积和授权限制未纳入仓库。

该数据源自龙口市公交公司，字段包括：`CONSUMEDATETIME`（刷卡时间）、`LINECODE`（线路）、`BUSCODE`（车辆）、`BUSSTOP`（上车站点）、`TASK`（上下行）、`CARDTYPENAME`（卡类型）。原始数据已脱敏，不含卡号、身份证等个人身份信息。

若需从原始数据重跑 ETL，可参考 `src/01_data_preprocessing.py` 的处理逻辑，该脚本说明了如何从原始刷卡记录聚合出本仓库的 `aggregated_data_full.csv`。
