import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from config import AGGREGATED_CSV, MODEL_DIR, ensure_dirs, require_file

ensure_dirs()

print('模型训练')

# 1.读取聚合数据
print('\n1. 读取数据...')
df = pd.read_csv(require_file(AGGREGATED_CSV))
print('数据形状：', df.shape)

# 2.添加新特征
print('\n2. 特征工程.')
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['weekday'] >= 5).astype(int)

# 3.准备特征和目标
features = ['hour', 'LINECODE', 'temp_celsius', 'tp', 'is_holiday', 'weekday', 'month']
X = df[features]
y = df['ride_count']

# 4.划分数据
print('\n3. 数据划分（8:1:1）...')
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('训练集大小：', X_train.shape)
print('验证集大小：', X_val.shape)
print('测试集大小：', X_test.shape)

# 模型1：线性回归
print('模型1：线性回归')
print('='*60)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('R2:', round(r2_score(y_test, y_pred_lr), 4))
print('MAE:', round(mean_absolute_error(y_test, y_pred_lr), 2))

# 模型2：随机森林
print('模型2：随机森林')

# 减少过拟合
rf = RandomForestRegressor(
    n_estimators=100,      # 100棵树
    max_depth=10,          # 限制最大深度
    min_samples_leaf=5,    # 叶子节点最少样本数
    random_state=42, 
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 验证集评估
y_pred_val = rf.predict(X_val)
r2_val = r2_score(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

# 测试集评估
y_pred_test = rf.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

# 训练集评估
y_pred_train = rf.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)

print('训练集 R2:', round(r2_train, 4))
print('验证集 R2:', round(r2_val, 4))
print('测试集 R2:', round(r2_test, 4))
print('验证集 MAE:', round(mae_val, 2))
print('测试集 MAE:', round(mae_test, 2))

# 过拟合检测
overfit_gap = r2_train - r2_val
print('\n过拟合检测：')
print('  训练集-验证集差异：%.4f' % overfit_gap)
if overfit_gap > 0.1:
    print('结论：存在过拟合')
else:
    print('结论：无明显过拟合')

# 特征重要性
print('\n特征重要性：')
for feat, imp in zip(features, rf.feature_importances_):
    print('  %s: %.4f' % (feat, imp))

# 车辆数预测
print('车辆数预测（35人/车基准）')

PASSENGERS_PER_BUS = 35

print('基准参数：')
print('每车载客：', PASSENGERS_PER_BUS, '人')

# 单线路预测示例
print('\n预测示例：线路6，工作日，各时段')
predict_data = pd.DataFrame({
    'hour': range(5, 20),
    'LINECODE': [6] * 15,
    'temp_celsius': [10] * 15,
    'tp': [0] * 15,
    'is_holiday': [0] * 15,
    'weekday': [2] * 15,
    'month': [11] * 15
})

predictions = rf.predict(predict_data)

print('\n时段 | 预测乘车量 | 建议车辆数')
print('-'*60)
for hour, pred in zip(range(5, 20), predictions):
    buses_needed = max(1, int(np.ceil(pred / PASSENGERS_PER_BUS)))
    print('%d:00-%d:59  |  %6.0f人  |  %d辆' % (hour, hour, pred, buses_needed))


print('模型对比')
print('线性回归 R2:', round(r2_score(y_test, y_pred_lr), 4))
print('随机森林 R2:', round(r2_test, 4))
print('过拟合差：', round(overfit_gap, 4))
print('\n推荐使用优化后的随机森林模型进行预测！')

# 保存模型
print('保存模型')
model_dir = MODEL_DIR
joblib.dump(rf, model_dir / 'rf_model.pkl')
joblib.dump(lr, model_dir / 'lr_model.pkl')
joblib.dump(features, model_dir / 'feature_info.pkl')
print(f'模型已保存至 {model_dir}: rf_model.pkl, lr_model.pkl, feature_info.pkl')