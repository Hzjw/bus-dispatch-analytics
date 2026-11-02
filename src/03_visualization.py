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

from config import AGGREGATED_CSV, FIGURE_DIR, MODEL_DIR, ensure_dirs, require_file

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


print('可视化分析')

# 1. 读取数据
print('\n1. 读取数据...')
df = pd.read_csv(require_file(AGGREGATED_CSV))

# 添加特征
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# 2. 加载三层分层模型
print('\n2. 加载三层分层模型...')
rf_workday = joblib.load(MODEL_DIR / 'rf_model_workday.pkl')
rf_weekend = joblib.load(MODEL_DIR / 'rf_model_weekend.pkl')
rf_holiday = joblib.load(MODEL_DIR / 'rf_model_holiday.pkl')
features = ['hour', 'LINECODE', 'temp_celsius', 'tp', 'is_holiday', 'weekday', 'month']
print('  三层分层模型已加载（工作日/周末/节假日）')

# 三层分层预测函数
def predict_stratified(X_df):
    """根据 is_holiday 和 weekday 自动路由到对应模型"""
    y_pred = np.zeros(len(X_df))
    mask_hol = X_df['is_holiday'] == 1
    mask_wknd = (X_df['is_holiday'] == 0) & (X_df['weekday'] >= 5)
    mask_wkdy = (X_df['is_holiday'] == 0) & (X_df['weekday'] < 5)
    if mask_hol.sum() > 0:
        y_pred[mask_hol] = rf_holiday.predict(X_df[mask_hol])
    if mask_wknd.sum() > 0:
        y_pred[mask_wknd] = rf_weekend.predict(X_df[mask_wknd])
    if mask_wkdy.sum() > 0:
        y_pred[mask_wkdy] = rf_workday.predict(X_df[mask_wkdy])
    return y_pred

# 准备数据并划分（与训练时保持一致：random_state=42）
X = df[features]
y = df['ride_count']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 分层预测
y_pred = predict_stratified(X_test)
y_pred_train = predict_stratified(X_train)
y_pred_val = predict_stratified(X_val)

# 计算整体R2
r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)
r2_test = r2_score(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)

print('  训练集R2: %.4f' % r2_train)
print('  验证集R2: %.4f' % r2_val)
print('  测试集R2: %.4f' % r2_test)
print('  测试集MAE: %.2f' % mae_test)
print('  过拟合差: %.4f' % (r2_train - r2_val))

# 分层偏差统计
print('\n  分层预测偏差:')
for label, mask_fn in [
    ('工作日', (X_test['is_holiday'] == 0) & (X_test['weekday'] < 5)),
    ('周末',   (X_test['is_holiday'] == 0) & (X_test['weekday'] >= 5)),
    ('节假日', X_test['is_holiday'] == 1),
]:
    if mask_fn.sum() > 0:
        bias = (y_pred[mask_fn] - y_test.values[mask_fn]).mean()
        print('    %s: 偏差 %+.2f (样本 %d)' % (label, bias, mask_fn.sum()))

# 图1：三层模型特征重要性对比
print('\n3. 生成特征重要性图...')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
feat_names = ['小时', '线路', '气温', '降水量', '节假日', '星期', '月份']
model_labels = ['工作日模型', '周末模型', '节假日模型']
model_importances = [
    rf_workday.feature_importances_,
    rf_weekend.feature_importances_,
    rf_holiday.feature_importances_,
]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#87CEEB']

for i, (ax, label, imps) in enumerate(zip(axes, model_labels, model_importances)):
    bars = ax.barh(feat_names, imps, color=colors)
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.set_xlabel('重要性', fontsize=11)
    for bar, imp in zip(bars, imps):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                '%.3f' % imp, va='center', fontsize=9)
fig.suptitle('三层分层模型 - 特征重要性对比', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  特征重要性图已保存')

# 图2：实际预测值
print('\n4. 生成实际vs预测图...')
fig, ax = plt.subplots(figsize=(10, 6))
sample_idx = np.random.choice(len(y_test), 200, replace=False)
ax.scatter(y_test.values[sample_idx], y_pred[sample_idx], alpha=0.5, color='#4ECDC4', s=30)
max_val = max(y_test.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')
ax.set_xlabel('实际乘车次数', fontsize=12)
ax.set_ylabel('预测乘车次数', fontsize=12)
ax.set_title('实际值 vs 预测值（R2=%.4f）' % r2_test, fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print('实际vs预测图已保存')

# 图3：各时段预测对比
print('\n5. 生成时段对比图...')
fig, ax = plt.subplots(figsize=(10, 6))
test_results = pd.DataFrame({'hour': X_test['hour'].values, 'actual': y_test.values, 'predicted': y_pred})
hourly_stats = test_results.groupby('hour').agg({'actual': 'mean', 'predicted': 'mean'}).reset_index()
ax.plot(hourly_stats['hour'], hourly_stats['actual'], 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='实际值')
ax.plot(hourly_stats['hour'], hourly_stats['predicted'], 's--', color='#4ECDC4', linewidth=2, markersize=8, label='预测值')
ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('平均乘车次数', fontsize=12)
ax.set_title('各时段平均乘车量对比（实际 vs 预测）', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'hourly_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print('  时段对比图已保存')

# 图4：误差分布
print('\n6. 生成误差分布图...')
fig, ax = plt.subplots(figsize=(10, 6))
errors = y_test.values - y_pred
ax.hist(errors, bins=50, color='#45B7D1', edgecolor='white', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('预测误差', fontsize=12)
ax.set_ylabel('频次', fontsize=12)
ax.set_title('预测误差分布', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('  误差分布图已保存')

# 图5：各线路预测效果
print('\n7. 生成线路预测效果图...')
fig, ax = plt.subplots(figsize=(10, 6))
top_lines = df['LINECODE'].value_counts().head(10).index.tolist()
line_results = []
for line in top_lines:
    mask = X_test['LINECODE'] == line
    if mask.sum() > 0:
        r2 = r2_score(y_test.values[mask], y_pred[mask])
        line_results.append({'线路': line, 'R2': r2})
line_df = pd.DataFrame(line_results)
colors5 = plt.cm.RdYlGn(line_df['R2'] / line_df['R2'].max())
bars5 = ax.bar(line_df['线路'].astype(str), line_df['R2'], color=colors5)
ax.set_xlabel('线路编号', fontsize=12)
ax.set_ylabel('R2值', fontsize=12)
ax.set_title('各线路预测效果（R2值）', fontsize=14, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', label='良好阈值(0.7)')
ax.legend()
for bar, r2 in zip(bars5, line_df['R2']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '%.2f' % r2, ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'route_r2.png', dpi=150, bbox_inches='tight')
plt.close()
print('  线路预测效果图已保存')

# 图6：车辆预测图
print('\n8. 生成车辆预测图...')
fig, ax1 = plt.subplots(figsize=(12, 6))

predict_data = pd.DataFrame({
    'hour': range(5, 20),
    'LINECODE': [6] * 15,
    'temp_celsius': [10] * 15,
    'tp': [0] * 15,
    'is_holiday': [0] * 15,
    'weekday': [2] * 15,
    'month': [11] * 15
})
predictions = predict_stratified(predict_data)

PASSENGERS_PER_BUS = 35
buses_needed = [max(1, int(np.ceil(p / PASSENGERS_PER_BUS))) for p in predictions]

x = np.arange(len(range(5, 20)))
ax1.bar(x, predictions, color='#4ECDC4', alpha=0.7, label='预测乘车量')
ax1_twin = ax1.twinx()
ax1_twin.plot(x, buses_needed, 'ro-', linewidth=2, markersize=8, label='建议车辆数')

ax1.set_xlabel('小时', fontsize=12)
ax1.set_ylabel('预测乘车量', fontsize=12)
ax1_twin.set_ylabel('建议车辆数', fontsize=12)
ax1.set_title('线路6各时段预测乘车量与建议车辆数', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(range(5, 20))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'route6_bus_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print('  车辆预测图已保存')

# 图7：固定间隔运力问题
print('\n9. 生成固定间隔运力问题图...')
fig, ax = plt.subplots(figsize=(12, 6))

# 直接使用聚合表：ride_count 即该「日期 × 小时 × 线路」的刷卡乘次，
# 等价于原始流水按小时 size() 统计，无需加载 1.2 GB 原始数据
day_mask = df['date'].dt.strftime('%Y-%m-%d') == '2023-11-12'
line6_day = df[(df['LINECODE'] == 6) & day_mask]
hourly_rides = line6_day.groupby('hour')['ride_count'].sum()

hours = list(range(5, 20))
actual_rides = [hourly_rides.get(h, 0) for h in hours]
# 固定间隔：取全天各时段平均需求，统一分配运力（模拟传统固定发车间隔模式）
PASSENGERS_PER_BUS = 35
active_hours = [r for r in actual_rides if r > 0]
avg_rides_per_hour = sum(active_hours) / len(active_hours)
fixed_buses_per_hour = max(1, int(np.ceil(avg_rides_per_hour / PASSENGERS_PER_BUS)))
fixed_capacity_per_hour = fixed_buses_per_hour * PASSENGERS_PER_BUS
fixed_capacity = [fixed_capacity_per_hour] * len(hours)

x = np.arange(len(hours))
width = 0.35

ax.bar(x - width/2, actual_rides, width, label='实际乘车量', color='#4ECDC4')
ax.bar(x + width/2, fixed_capacity, width, label='固定间隔运力(%d人/小时)' % fixed_capacity_per_hour, color='#FF6B6B', alpha=0.5)

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('人数', fontsize=12)
ax.set_title('固定间隔运力 vs 实际需求（线路6，2023-11-12）', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xticks(x)
ax.set_xticklabels(hours)

total_rides = sum(actual_rides)
total_fixed = fixed_capacity_per_hour * len(hours)
utilization = total_rides / total_fixed * 100

ax.text(0.02, 0.98, '运力利用率: %.1f%%  浪费: %d人（%d辆/时×%d人）' % (utilization, total_fixed - total_rides, fixed_buses_per_hour, PASSENGERS_PER_BUS), 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'fixed_interval_problem.png', dpi=150, bbox_inches='tight')
plt.close()
print('  固定间隔问题图已保存')

# 图8：动态间隔建议
print('\n10. 生成动态间隔建议图...')
fig, ax = plt.subplots(figsize=(12, 6))

PASSENGERS_PER_BUS = 35

dynamic_capacity = []
for rides in actual_rides:
    if rides > 0:
        buses = max(1, int(np.ceil(rides / PASSENGERS_PER_BUS)))
        capacity = buses * PASSENGERS_PER_BUS
    else:
        capacity = PASSENGERS_PER_BUS
    dynamic_capacity.append(capacity)

ax.bar(x - width/2, actual_rides, width, label='实际乘车量', color='#4ECDC4')
ax.bar(x + width/2, dynamic_capacity, width, label='动态间隔运力', color='#FFA500', alpha=0.7)

ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('人数', fontsize=12)
ax.set_title('动态间隔运力 vs 实际需求（高峰期增加运力）', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xticks(x)
ax.set_xticklabels(hours)

total_dynamic = sum(dynamic_capacity)
dynamic_utilization = total_rides / total_dynamic * 100
savings = total_fixed - total_dynamic

ax.text(0.02, 0.98, '运力利用率: %.1f%%  节省: %d人' % (dynamic_utilization, savings), 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'dynamic_interval.png', dpi=150, bbox_inches='tight')
plt.close()
print('  动态间隔建议图已保存')

# 图9：调度优化效果对比
print('\n11. 生成调度优化效果对比图...')
fig, ax = plt.subplots(figsize=(12, 6))

fixed_utilization = [ride / 210 * 100 for ride in actual_rides]
dynamic_utilization = [ride / cap * 100 if cap > 0 else 0 for ride, cap in zip(actual_rides, dynamic_capacity)]

ax.plot(hours, fixed_utilization, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='固定间隔利用率')
ax.plot(hours, dynamic_utilization, 's-', color='#4ECDC4', linewidth=2, markersize=8, label='动态间隔利用率')
ax.set_xlabel('小时', fontsize=12)
ax.set_ylabel('运力利用率(%)', fontsize=12)
ax.set_title('固定间隔 vs 动态间隔运力利用率对比', fontsize=14, fontweight='bold')
ax.legend()
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100%利用率')
ax.set_xticks(hours)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / 'utilization_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  调度优化效果对比图已保存')

# 图10：天气调度建议
print('\n12. 生成天气调度建议图...')

# 从数据中计算天气影响
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
weather_stats = df.groupby('weather_type').agg({'ride_count': ['mean', 'count']})
weather_stats.columns = ['avg_rides', 'count']
base_rides = weather_stats.loc['舒适', 'avg_rides']
weather_stats['change_pct'] = ((weather_stats['avg_rides'] - base_rides) / base_rides * 100).round(1)
weather_stats['dispatch_adjustment'] = weather_stats['change_pct'].round(0)
weather_sorted = weather_stats.sort_values('dispatch_adjustment')

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4ECDC4' if v >= 0 else '#FF6B6B' for v in weather_sorted['dispatch_adjustment']]
bars = ax.bar(weather_sorted.index, weather_sorted['dispatch_adjustment'], color=colors)
ax.set_xlabel('天气类型', fontsize=12)
ax.set_ylabel('发车调整比例(%)', fontsize=12)
ax.set_title('不同天气条件下的发车调整建议', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
for bar, pct in zip(bars, weather_sorted['dispatch_adjustment']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1 if pct >= 0 else bar.get_height() - 3,
            '%+.0f%%' % pct, ha='center', fontsize=10, fontweight='bold')
ax.text(0.02, 0.98, '绿色=增加发车  红色=减少发车',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'weather_dispatch_advice.png', dpi=150, bbox_inches='tight')
plt.close()
print('天气调度建议图已保存')

print('所有可视化图表已生成完成！')
