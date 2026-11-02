import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from config import AGGREGATED_CSV, MODEL_DIR, ensure_dirs, require_file

ensure_dirs()

PASSENGERS_PER_BUS = 35

print('三层分层建模')

# 读取数据
print('\n1. 读取数据...')
df = pd.read_csv(require_file(AGGREGATED_CSV))
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
print('总数据量：', df.shape)

# 三层层拆分
features = ['hour', 'LINECODE', 'temp_celsius', 'tp', 'is_holiday', 'weekday', 'month']

df_holiday = df[df['is_holiday'] == 1].copy()
df_weekend = df[(df['is_holiday'] == 0) & (df['weekday'] >= 5)].copy()
df_workday = df[(df['is_holiday'] == 0) & (df['weekday'] < 5)].copy()

total = df.shape[0]
print(f'\n节假日数据：{df_holiday.shape[0]:>8} 条 ({df_holiday.shape[0]/total*100:.1f}%)')
print(f'周末数据：  {df_weekend.shape[0]:>8} 条 ({df_weekend.shape[0]/total*100:.1f}%)')
print(f'工作日数据：{df_workday.shape[0]:>8} 条 ({df_workday.shape[0]/total*100:.1f}%)')

# 训练函数
def train_model(df_subset, label, params=None):
    if params is None:
        params = dict(n_estimators=100, max_depth=10, min_samples_leaf=5,
                      random_state=42, n_jobs=-1)

    X = df_subset[features]
    y = df_subset['ride_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f'\n{"="*40}')
    print(f'{label} 模型')
    print(f'{"="*40}')
    print(f'训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条')
    print(f'R² 训练集: {r2_train:.4f}')
    print(f'R² 测试集: {r2_test:.4f}')
    print(f'MAE 测试集: {mae_test:.2f}')
    print(f'过拟合差: {r2_train - r2_test:.4f}')
    print(f'\n特征重要性:')
    for feat, imp in zip(features, rf.feature_importances_):
        print(f'  {feat}: {imp:.4f}')

    return rf, r2_test, mae_test

# 工作日模型
rf_workday, r2_workday, mae_workday = train_model(df_workday, '工作日')

# 周末模型
rf_weekend, r2_weekend, mae_weekend = train_model(df_weekend, '周末')

# 节假日模型（样本较少，减少深度防过拟合）
holiday_params = dict(n_estimators=80, max_depth=8, min_samples_leaf=3,
                      random_state=42, n_jobs=-1)
rf_holiday, r2_holiday, mae_holiday = train_model(df_holiday, '节假日', holiday_params)

# 各层实际均值 vs 预测均值对比
print('对比分析：各层实际客流量与预测均值')

# 加载原模型
rf_original = joblib.load(MODEL_DIR / 'rf_model.pkl')

for label, df_sub, rf_sub in [('工作日', df_workday, rf_workday),
                                ('周末', df_weekend, rf_weekend),
                                ('节假日', df_holiday, rf_holiday)]:
    X = df_sub[features]
    y = df_sub['ride_count']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    actual_mean = y_test.mean()
    orig_pred = rf_original.predict(X_test)
    orig_mean = orig_pred.mean()
    strat_pred = rf_sub.predict(X_test)
    strat_mean = strat_pred.mean()
    strat_r2 = r2_score(y_test, strat_pred)
    orig_r2 = r2_score(y_test, orig_pred)

    print(f'\n{label}:')
    print(f'实际均值: {actual_mean:.2f}')
    print(f'原模型预测均值: {orig_mean:.2f} (偏差 {orig_mean - actual_mean:+.2f})')
    print(f'分层模型预测均值: {strat_mean:.2f} (偏差 {strat_mean - actual_mean:+.2f})')
    print(f'原模型 R²: {orig_r2:.4f} → 分层模型 R²: {strat_r2:.4f}')

# 整体对比
print('整体对比（三层分层与原模型）')

df_all = df[features]
y_all = df['ride_count']
_, X_all_test, _, y_all_test = train_test_split(df_all, y_all, test_size=0.2, random_state=42)

# 三层分层预测
y_all_strat = np.zeros(len(X_all_test))
mask_hol = X_all_test['is_holiday'] == 1
mask_wknd = (X_all_test['is_holiday'] == 0) & (X_all_test['weekday'] >= 5)
mask_wkdy = (X_all_test['is_holiday'] == 0) & (X_all_test['weekday'] < 5)

if mask_hol.sum() > 0:
    y_all_strat[mask_hol] = rf_holiday.predict(X_all_test[mask_hol])
if mask_wknd.sum() > 0:
    y_all_strat[mask_wknd] = rf_weekend.predict(X_all_test[mask_wknd])
if mask_wkdy.sum() > 0:
    y_all_strat[mask_wkdy] = rf_workday.predict(X_all_test[mask_wkdy])

strat_r2_all = r2_score(y_all_test, y_all_strat)
strat_mae_all = mean_absolute_error(y_all_test, y_all_strat)

y_all_orig = rf_original.predict(X_all_test)
orig_r2_all = r2_score(y_all_test, y_all_orig)
orig_mae_all = mean_absolute_error(y_all_test, y_all_orig)

print(f'\n原模型: R²={orig_r2_all:.4f}, MAE={orig_mae_all:.2f}')
print(f'三层分层: R²={strat_r2_all:.4f}, MAE={strat_mae_all:.2f}')
print(f'改善: R² {strat_r2_all - orig_r2_all:+.4f}, MAE {strat_mae_all - orig_mae_all:+.2f}')

# 6. 保存模型
print('保存三层分层模型')

model_dir = MODEL_DIR
joblib.dump(rf_workday, model_dir / 'rf_model_workday.pkl')
joblib.dump(rf_weekend, model_dir / 'rf_model_weekend.pkl')
joblib.dump(rf_holiday, model_dir / 'rf_model_holiday.pkl')
print('工作日模型已保存: rf_model_workday.pkl')
print('周末模型已保存:   rf_model_weekend.pkl')
print('节假日模型已保存: rf_model_holiday.pkl')
print(f'（保存位置：{model_dir}）')

print('\n三层分层建模完成！')