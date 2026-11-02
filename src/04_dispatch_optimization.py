import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import (
    AGGREGATED_CSV,
    DISPATCH_FIGURE_DIR,
    MODEL_DIR,
    PASSENGERS_PER_BUS,
    ensure_dirs,
    require_file,
)

ensure_dirs()

# 配置中文字体
def setup_chinese_font():
    font_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    for font_path in font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

print('调度优化分析')

# 1.读取数据
print('\n1. 读取数据...')
df = pd.read_csv(require_file(AGGREGATED_CSV))

# 添加特征工程
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# 2.加载已训练模型
print('\n2. 加载已训练模型...')
rf = joblib.load(MODEL_DIR / 'rf_model.pkl')
lr = joblib.load(MODEL_DIR / 'lr_model.pkl')
features = ['hour', 'LINECODE', 'temp_celsius', 'tp', 'is_holiday', 'weekday', 'month']
print('  模型已加载，特征: %s' % features)

# 3.计算当前发车间隔（基于实际数据，按线路分别统计后取均值）
print('\n3. 计算当前发车间隔...')
# 聚合表中的 bus_count 即该「日期 × 小时 × 线路」的运营车辆数（原始 BUSCODE 唯一计数），
# 先按线路 × 小时取日均，再对线路取均值，得到各时段的平均配车规模。
# 这样无需加载 1.2 GB 原始流水。
avg_buses_per_hour = (
    df.groupby(['LINECODE', 'hour'])['bus_count'].mean()
      .groupby('hour').mean()
)
dispatch_interval = 60 / avg_buses_per_hour

print('\n各时段当前发车间隔（各线路均值）：')
for hour in sorted(avg_buses_per_hour.index):
    n = avg_buses_per_hour[hour]
    interval = dispatch_interval[hour]
    print(f'  {hour:2d}:00 - 平均{n:.1f}辆车/线路, 约{interval:.1f}分钟/班')

# 4.天气影响分析
print('\n4. 天气影响分析...')

# 添加天气类型分类
def classify_weather(row):
    if row['tp'] > 0 and row['temp_celsius'] <= 0:
        return '雪天'
    elif row['tp'] > 0 and row['temp_celsius'] > 0:
        return '雨天'
    elif row['tp'] == 0 and row['temp_celsius'] >= 30:
        return '炎热'
    elif row['tp'] == 0 and row['temp_celsius'] < 5:
        return '寒冷'
    else:
        return '舒适'

df['weather_type'] = df.apply(classify_weather, axis=1)

weather_stats = df.groupby('weather_type').agg({
    'ride_count': ['mean', 'count']
}).reset_index()
weather_stats.columns = ['weather_type', 'avg_rides', 'count']

# 计算较基准（舒适天气）的变化比例
base_rides = weather_stats[weather_stats['weather_type'] == '舒适']['avg_rides'].values[0]
weather_stats['change_pct'] = ((weather_stats['avg_rides'] - base_rides) / base_rides * 100).round(1)
weather_stats['dispatch_adjustment'] = weather_stats['change_pct'].round(0)

print('\n天气对客流的影响：')
for _, row in weather_stats.iterrows():
    print(f'  {row["weather_type"]}: 日均{row["avg_rides"]:.0f}次, 变化{row["change_pct"]:+.1f}%')

# 5.生成天气调度建议图
print('\n5. 生成天气调度建议图...')
fig, ax = plt.subplots(figsize=(10, 6))

# 按变化比例排序
weather_sorted = weather_stats.sort_values('dispatch_adjustment')
colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in weather_sorted['dispatch_adjustment']]
bars = ax.bar(weather_sorted['weather_type'], weather_sorted['dispatch_adjustment'], color=colors)

ax.set_xlabel('天气类型', fontsize=12)
ax.set_ylabel('发车调整比例(%)', fontsize=12)
ax.set_title('不同天气条件下的发车调整建议', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

for bar, pct in zip(bars, weather_sorted['dispatch_adjustment']):
    ax.text(bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 1 if pct >= 0 else bar.get_height() - 3,
            f'{pct:+.0f}%', ha='center', fontsize=10, fontweight='bold')

ax.text(0.02, 0.98, '绿色=增加发车  红色=减少发车', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'weather_dispatch_advice.png', dpi=150, bbox_inches='tight')
plt.close()
print('  天气调度建议图已保存')

# 6.调度建议：模型预测客流后，直接按公式换算
#    建议发车数 = ceil(预测客流量 / 35人/车)
#    此步骤为纯公式计算，无需单独可视化，在报告中以文字说明即可
print('\n6. 调度建议换算逻辑：')
print('  建议发车数 = ceil(模型预测客流量 / 35人/车)')
print('  该换算直接嵌入模型预测流程，无需独立图表展示')

# 7.准备模型预测数据
print('\n7. 准备预测数据...')
X = df[features]
y = df['ride_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_pred = rf.predict(X_test)

# 各时段平均乘车量
print('\n8. 生成各时段平均乘车量图...')
fig, ax = plt.subplots(figsize=(10, 6))
hourly_avg = df.groupby('hour')['ride_count'].mean()
ax.bar(hourly_avg.index, hourly_avg.values, color='#4ECDC4')
ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('平均乘车量', fontsize=12)
ax.set_title('各时段平均乘车量', fontsize=14, fontweight='bold')
ax.axhline(y=hourly_avg.mean(), color='red', linestyle='--', label='平均值')
ax.legend()
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'hourly_volume.png', dpi=150, bbox_inches='tight')
plt.close()
print('各时段平均乘车量图已保存')

# 不同天气类型平均乘车量
print('\n9. 生成天气类型对比图...')
fig, ax = plt.subplots(figsize=(10, 6))
weather_avg = df.groupby('weather_type')['ride_count'].mean().sort_values(ascending=True)
colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
ax.barh(weather_avg.index, weather_avg.values, color=colors_list[:len(weather_avg)])
ax.set_xlabel('平均乘车量', fontsize=12)
ax.set_title('不同天气类型平均乘车量', fontsize=14, fontweight='bold')
for i, v in enumerate(weather_avg.values):
    ax.text(v + 10, i, f'{v:.0f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'weather_volume.png', dpi=150, bbox_inches='tight')
plt.close()
print('  天气类型对比图已保存')

# 工作日与节假日
print('\n10. 生成工作日与节假日图...')
fig, ax = plt.subplots(figsize=(8, 6))
holiday_avg = df.groupby('is_holiday')['ride_count'].mean()
ax.bar(['工作日', '节假日'], holiday_avg.values, color=['#4ECDC4', '#FF6B6B'])
ax.set_ylabel('平均乘车量', fontsize=12)
ax.set_title('工作日与节假日平均乘车量', fontsize=14, fontweight='bold')
for i, v in enumerate(holiday_avg.values):
    ax.text(i, v + 10, f'{v:.0f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'workday_vs_holiday.png', dpi=150, bbox_inches='tight')
plt.close()
print('  工作日与节假日图已保存')

# 预测与实际对比
print('\n11. 生成预测与实际对比图...')
fig, ax = plt.subplots(figsize=(10, 6))
sample_idx = np.random.choice(len(y_test), 200, replace=False)
ax.scatter(y_test.values[sample_idx], y_pred[sample_idx], alpha=0.5, color='#4ECDC4', s=30)
max_val = max(y_test.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')
ax.set_xlabel('实际乘车量', fontsize=12)
ax.set_ylabel('预测乘车量', fontsize=12)
ax.set_title('预测与实际对比', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('实际与预测对比图已保存')

# 模型R²值对比
print('\n12. 生成模型R²值对比图...')
fig, ax = plt.subplots(figsize=(8, 6))
lr_r2 = lr.score(X_test, y_test)
rf_r2 = rf.score(X_test, y_test)
ax.bar(['线性回归', '随机森林'], [lr_r2, rf_r2], color=['#FF6B6B', '#4ECDC4'])
ax.set_ylabel('R²值', fontsize=12)
ax.set_title('模型R²值对比', fontsize=14, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', label='良好阈值')
ax.legend()
# 标注数值
for i, (v, label) in enumerate(zip([lr_r2, rf_r2], [lr_r2, rf_r2])):
    ax.text(i, v + 0.02, '%.4f' % v, ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'model_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  模型R²值对比图已保存')

# 特征重要性
print('\n13. 生成特征重要性图...')
fig, ax = plt.subplots(figsize=(10, 6))
feat_names = ['小时', '线路', '气温', '降水量', '节假日', '星期', '月份']
importances = rf.feature_importances_
colors_feat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#87CEEB']
bars = ax.barh(feat_names, importances, color=colors_feat)
ax.set_xlabel('重要性', fontsize=12)
ax.set_title('随机森林特征重要性', fontsize=14, fontweight='bold')
for bar, imp in zip(bars, importances):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, '%.3f' % imp, va='center', fontsize=10)
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'feature_importance_dispatch.png', dpi=150, bbox_inches='tight')
plt.close()
print('  特征重要性图已保存')

# 预测误差分布
print('\n14. 生成预测误差分布图...')
fig, ax = plt.subplots(figsize=(10, 6))
errors = y_test.values - y_pred
ax.hist(errors, bins=50, color='#45B7D1', edgecolor='white', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('预测误差', fontsize=12)
ax.set_ylabel('频次', fontsize=12)
ax.set_title('预测误差分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(DISPATCH_FIGURE_DIR / 'error_distribution_dispatch.png', dpi=150, bbox_inches='tight')
plt.close()

print('预测误差分布图已保存')

print('调度优化分析图表已生成')
