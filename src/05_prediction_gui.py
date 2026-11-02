import sys
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QComboBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QGridLayout, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import AGGREGATED_CSV, MODEL_DIR


def setup_chinese_font():
    font_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    for font_path in font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

PASSENGERS_PER_BUS = 35
FEATURES = ['hour', 'LINECODE', 'temp_celsius', 'tp', 'is_holiday', 'weekday', 'month']
WEATHER_PRESETS = {
    '舒适': (18, 0),
    '炎热': (33, 0),
    '寒冷': (0, 0),
    '雨天': (15, 5),
    '雪天': (-3, 3),
}
LINE_COLORS = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#87CEEB']


STYLE_SHEET = """
QMainWindow {
    background-color: #f0f4f8;
}
QLabel#title {
    font-size: 30px;
    font-weight: bold;
    color: #1a365d;
    padding: 8px;
}
QLabel#subtitle {
    font-size: 24px;
    color: #718096;
}
QGroupBox {
    color: #1a365d;
    border: 2px solid #4ECDC4;
    border-radius: 8px;
    margin-top: 36px;
    padding-top: 36px;
    background-color: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 2px;
    padding: 2px 10px;
    color: #1a365d;
    background-color: white;
    border-radius: 4px;
}
QComboBox {
    font-size: 24px;
    padding: 8px 12px;
    border: 1px solid #cbd5e0;
    border-radius: 4px;
    background: white;
    min-width: 120px;
}
QComboBox:hover {
    border-color: #4ECDC4;
}
QPushButton {
    font-size: 24px;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 5px;
    color: white;
    background-color: #4ECDC4;
    border: none;
}
QPushButton:hover {
    background-color: #45b7aa;
}
QPushButton:pressed {
    background-color: #3ca090;
}
QTableWidget {
    font-size: 24px;
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    gridline-color: #edf2f7;
}
QTableWidget::item {
    padding: 8px;
}
QHeaderView::section {
    background-color: #f7fafc;
    font-size: 24px;
    font-weight: bold;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #e2e8f0;
}
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: white;
}
QTabBar::tab {
    font-size: 24px;
    padding: 10px 24px;
    margin-right: 2px;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    background: #edf2f7;
    color: #4a5568;
}
QTabBar::tab:selected {
    background: white;
    color: #2d3748;
    font-weight: bold;
    border-bottom: 2px solid #4ECDC4;
}
QLabel {
    font-size: 24px;
}
"""

# ============================================================
# 指标卡片
# ============================================================
class MetricCard(QFrame):
    """模型指标卡片"""
    def __init__(self, title, value, color='#4ECDC4'):
        super().__init__()
        self.setObjectName("metricCard")
        self.setStyleSheet("""
            QFrame#metricCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
        """)
        self.setMinimumHeight(100)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 8, 12, 8)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Normal))
        title_label.setStyleSheet("color: #718096;")
        title_label.setAlignment(Qt.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        value_label.setStyleSheet(f"color: {color};")
        value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)


# ============================================================
# Matplotlib 画布
# ============================================================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.tight_layout(pad=3)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fig.tight_layout(pad=3)
        self.draw_idle()


# ============================================================
# 主窗口
# ============================================================
class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("公交客流智能预测调度系统")
        self.setMinimumSize(1200, 800)
        
        # 加载模型
        self.load_models()
        
        # 加载数据
        self.load_data()
        
        # 初始化UI
        self.init_ui()
        
        # 初始预测
        self.update_all()
        
        # 最大化窗口
        self.showMaximized()

    def load_models(self):
        self.rf_workday = joblib.load(MODEL_DIR / 'rf_model_workday.pkl')
        self.rf_weekend = joblib.load(MODEL_DIR / 'rf_model_weekend.pkl')
        self.rf_holiday = joblib.load(MODEL_DIR / 'rf_model_holiday.pkl')
        self.model_version = '三层分层(7维)'
        self.model_r2 = '0.8438'
        print("三层分层模型加载完成（工作日/周末/节假日）")
        self.lr = joblib.load(MODEL_DIR / 'lr_model.pkl')

    def load_data(self):
        df = pd.read_csv(AGGREGATED_CSV, nrows=2000)
        df['date'] = pd.to_datetime(df['date'])
        self.available_lines = sorted(df['LINECODE'].unique().tolist())
        self.top_lines = df['LINECODE'].value_counts().head(10).index.tolist()

    def init_ui(self):
        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(12, 6, 12, 6)

        # ---- 顶部标题栏 ----
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("公交客流智能预测调度系统")
        title.setObjectName("title")
        sub = QLabel("三层分层随机森林 | 工作日/周末/节假日独立预测 | 实时调度建议")
        sub.setObjectName("subtitle")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(sub)
        main_layout.addLayout(header)

        # ---- 指标卡片 ----
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(8)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        self.card_r2 = MetricCard("随机森林 R²", self.model_r2, "#4ECDC4")
        self.card_mae = MetricCard("平均误差 MAE", "12.12", "#45B7D1")
        self.card_features = MetricCard("输入特征", "7 维", "#96CEB4")
        self.card_data = MetricCard("训练数据", "419,045 条", "#FFEAA7")
        cards_layout.addWidget(self.card_r2)
        cards_layout.addWidget(self.card_mae)
        cards_layout.addWidget(self.card_features)
        cards_layout.addWidget(self.card_data)
        main_layout.addLayout(cards_layout)

        # ---- 主体分栏 ----
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：图表 + 表格
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # 图表区
        chart_group = QGroupBox("预测图表")
        chart_group.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = MplCanvas()
        chart_layout.addWidget(self.canvas)
        right_layout.addWidget(chart_group, stretch=3)
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setAlternatingRowColors(True)
        self.result_table.verticalHeader().setDefaultSectionSize(50)
        right_layout.addWidget(self.result_table, stretch=1)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 1)

    def create_control_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(350)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # 场景选择 Tab
        tabs = QTabWidget()
        
        fnt = QFont("Microsoft YaHei", 24, QFont.Bold)
        
        # ---- Tab 1: 全时段预测 ----
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        t1_layout.setSpacing(10)
        
        g1 = QGroupBox("线路选择")
        g1.setFont(fnt)
        g1l = QVBoxLayout(g1)
        self.line_combo = QComboBox()
        self.line_combo.addItems([str(l) for l in self.top_lines])
        self.line_combo.setCurrentText('6')
        g1l.addWidget(self.line_combo)
        t1_layout.addWidget(g1)
        
        g2 = QGroupBox("日期类型")
        g2.setFont(fnt)
        g2l = QVBoxLayout(g2)
        self.day_combo = QComboBox()
        self.day_combo.addItems(['工作日', '周末', '节假日'])
        self.day_combo.currentTextChanged.connect(self.on_day_changed)
        g2l.addWidget(self.day_combo)
        t1_layout.addWidget(g2)
        
        g3 = QGroupBox("天气类型")
        g3.setFont(fnt)
        g3l = QVBoxLayout(g3)
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(['舒适', '炎热', '寒冷', '雨天', '雪天'])
        g3l.addWidget(self.weather_combo)
        t1_layout.addWidget(g3)
        
        g4 = QGroupBox("月份")
        g4.setFont(fnt)
        g4l = QHBoxLayout(g4)
        self.month_combo = QComboBox()
        self.month_combo.addItems([str(m) for m in range(1, 13)])
        self.month_combo.setCurrentText('11')
        g4l.addWidget(QLabel("月"))
        g4l.addWidget(self.month_combo)
        t1_layout.addWidget(g4)
        
        t1_layout.addStretch()
        tabs.addTab(tab1, "全时段预测")
        
        # ---- Tab 2: 天气对比 ----
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        g2_1 = QGroupBox("固定参数")
        g2_1.setFont(fnt)
        g2_1l = QGridLayout(g2_1)
        self.w_line_combo = QComboBox()
        self.w_line_combo.addItems([str(l) for l in self.top_lines])
        self.w_line_combo.setCurrentText('6')
        self.w_hour_combo = QComboBox()
        self.w_hour_combo.addItems([f'{h}:00' for h in range(5, 21)])
        self.w_hour_combo.setCurrentText('8:00')
        g2_1l.addWidget(QLabel("线路:"), 0, 0)
        g2_1l.addWidget(self.w_line_combo, 0, 1)
        g2_1l.addWidget(QLabel("时段:"), 1, 0)
        g2_1l.addWidget(self.w_hour_combo, 1, 1)
        t2_layout.addWidget(g2_1)
        t2_layout.addStretch()
        tabs.addTab(tab2, "天气对比")
        
        # ---- Tab 3: 多线路对比 ----
        tab3 = QWidget()
        t3_layout = QVBoxLayout(tab3)
        g3_1 = QGroupBox("固定参数")
        g3_1.setFont(fnt)
        g3_1l = QGridLayout(g3_1)
        self.ml_hour_combo = QComboBox()
        self.ml_hour_combo.addItems([f'{h}:00' for h in range(5, 21)])
        self.ml_hour_combo.setCurrentText('8:00')
        self.ml_weather_combo = QComboBox()
        self.ml_weather_combo.addItems(['舒适', '炎热', '寒冷', '雨天', '雪天'])
        self.ml_day_combo = QComboBox()
        self.ml_day_combo.addItems(['工作日', '周末', '节假日'])
        g3_1l.addWidget(QLabel("时段:"), 0, 0)
        g3_1l.addWidget(self.ml_hour_combo, 0, 1)
        g3_1l.addWidget(QLabel("天气:"), 1, 0)
        g3_1l.addWidget(self.ml_weather_combo, 1, 1)
        g3_1l.addWidget(QLabel("日期:"), 2, 0)
        g3_1l.addWidget(self.ml_day_combo, 2, 1)
        t3_layout.addWidget(g3_1)
        t3_layout.addStretch()
        tabs.addTab(tab3, "多线路对比")
        
        layout.addWidget(tabs)
        
        # 更新按钮
        self.btn_update = QPushButton("刷新预测")
        self.btn_update.clicked.connect(self.update_all)
        layout.addWidget(self.btn_update)
        
        # 连接信号 — 所有控件的切换都即时刷新
        all_combos = [
            self.line_combo, self.day_combo, self.weather_combo, self.month_combo,
            self.w_line_combo, self.w_hour_combo,
            self.ml_hour_combo, self.ml_weather_combo, self.ml_day_combo,
        ]
        for combo in all_combos:
            combo.currentTextChanged.connect(self.update_all)
        tabs.currentChanged.connect(self.on_tab_changed)
        
        return panel

    def on_day_changed(self, text):
        self.update_all()

    def on_tab_changed(self, index):
        self.update_all()

    def get_current_tab(self):
        return self.findChild(QTabWidget).currentIndex()

    def make_input(self, hour, line, temp, tp, is_holiday, weekday, month):
        return pd.DataFrame([{
            'hour': hour, 'LINECODE': int(line),
            'temp_celsius': temp, 'tp': tp,
            'is_holiday': is_holiday, 'weekday': weekday, 'month': month,
        }])

    def predict(self, X):
        """根据 is_holiday 和 weekday 自动选择三层分层模型"""
        is_hol = X['is_holiday'].iloc[0]
        wkday = X['weekday'].iloc[0]
        if is_hol == 1:
            return self.rf_holiday.predict(X)
        elif wkday >= 5:
            return self.rf_weekend.predict(X)
        else:
            return self.rf_workday.predict(X)

    # ============================================================
    # 更新全部
    # ============================================================
    def update_all(self):
        tab = self.get_current_tab()
        if tab == 0:
            self.update_hourly()
        elif tab == 1:
            self.update_weather()
        else:
            self.update_multiline()

    # ============================================================
    # Tab 1: 全时段预测
    # ============================================================
    def update_hourly(self):
        line = int(self.line_combo.currentText())
        day_type = self.day_combo.currentText()
        weather = self.weather_combo.currentText()
        month = int(self.month_combo.currentText())
        
        # 参数映射
        if day_type == '工作日':
            is_holiday, weekday = 0, 2
        elif day_type == '周末':
            is_holiday, weekday = 0, 6
        else:
            is_holiday, weekday = 1, 6
        
        temp, tp = WEATHER_PRESETS[weather]
        
        # 预测
        hours = list(range(5, 20))
        X = pd.concat([self.make_input(h, line, temp, tp, is_holiday, weekday, month) for h in hours], ignore_index=True)
        preds = self.predict(X)
        buses = [max(1, int(np.ceil(p / PASSENGERS_PER_BUS))) for p in preds]
        intervals = [60 / b for b in buses]
        
        # 绘制柱状图
        self.canvas.ax.clear()
        colors = []
        for h in hours:
            if 7 <= h <= 9:
                colors.append('#FF6B6B')
            elif 14 <= h <= 16:
                colors.append('#FFA500')
            else:
                colors.append('#4ECDC4')
        bars = self.canvas.ax.bar(hours, preds, color=colors, alpha=0.85, edgecolor='white')
        
        # 标注车辆数
        for h, pred, bus in zip(hours, preds, buses):
            self.canvas.ax.text(h, pred + max(preds) * 0.02, f'{bus}辆',
                               ha='center', fontsize=24, fontweight='bold', color='#2d3748')
        
        self.canvas.ax.set_xlabel('小时', fontsize=24)
        self.canvas.ax.set_ylabel('预测乘车量（人）', fontsize=24)
        self.canvas.ax.set_title(f'线路{line} · {day_type} · {weather} · {month}月', fontsize=28, fontweight='bold')
        self.canvas.ax.set_xticks(hours)
        self.canvas.ax.set_xlim(4, 20)
        self.canvas.ax.tick_params(labelsize=24)
        self.canvas.ax.grid(axis='y', alpha=0.3)
        
        total_buses = sum(buses)
        self.canvas.ax.text(0.98, 0.95, f'全天建议投入: {total_buses} 辆',
                           transform=self.canvas.ax.transAxes, fontsize=24,
                           ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='#FFEAA7', alpha=0.8))
        
        self.canvas.fig.tight_layout(pad=3)
        self.canvas.draw()
        
        # 更新表格
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(['时段', '预测乘车量', '建议车辆数', '建议发车间隔', '时段标签'])
        self.result_table.setRowCount(len(hours))
        for i, (h, pred, bus, interval) in enumerate(zip(hours, preds, buses, intervals)):
            tag = ''
            if 7 <= h <= 9:
                tag = '早高峰'
            elif 14 <= h <= 16:
                tag = '晚高峰'
            elif 5 <= h <= 6:
                tag = '早间'
            elif 17 <= h <= 19:
                tag = '晚间'
            else:
                tag = '平峰'
            
            items = [
                QTableWidgetItem(f'{h:02d}:00'),
                QTableWidgetItem(f'{pred:.0f} 人'),
                QTableWidgetItem(f'{bus} 辆'),
                QTableWidgetItem(f'{interval:.1f} 分钟'),
                QTableWidgetItem(tag),
            ]
            for j, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                if tag == '早高峰':
                    item.setBackground(QColor('#FFF0F0'))
                elif tag == '晚高峰':
                    item.setBackground(QColor('#FFF5E6'))
                self.result_table.setItem(i, j, item)
        
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # ============================================================
    # Tab 2: 天气对比
    # ============================================================
    def update_weather(self):
        line = int(self.w_line_combo.currentText())
        hour = int(self.w_hour_combo.currentText().replace(':00', ''))
        
        weathers = ['舒适', '炎热', '寒冷', '雨天', '雪天']
        preds = []
        buses = []
        
        for weather in weathers:
            temp, tp = WEATHER_PRESETS[weather]
            X = self.make_input(hour, line, temp, tp, 0, 2, 11)
            pred = self.predict(X)[0]
            bus = max(1, int(np.ceil(pred / PASSENGERS_PER_BUS)))
            preds.append(pred)
            buses.append(bus)
        
        # 绘图
        self.canvas.ax.clear()
        comfort_pred = preds[0]
        colors = []
        for i, w in enumerate(weathers):
            pct = (preds[i] - comfort_pred) / comfort_pred * 100
            if pct >= 0:
                colors.append('#4ECDC4')
            else:
                colors.append('#FF6B6B')
        
        bars = self.canvas.ax.bar(weathers, preds, color=colors, alpha=0.85, edgecolor='white')
        
        for i, (w, pred, bus) in enumerate(zip(weathers, preds, buses)):
            pct = (pred - comfort_pred) / comfort_pred * 100
            label = f'{bus}辆\n({pct:+.1f}%)' if i > 0 else f'{bus}辆\n(基准)'
            self.canvas.ax.text(i, pred + max(preds) * 0.02, label,
                               ha='center', fontsize=24, fontweight='bold', color='#2d3748')
        
        self.canvas.ax.set_xlabel('天气类型', fontsize=24)
        self.canvas.ax.set_ylabel('预测乘车量（人）', fontsize=24)
        self.canvas.ax.set_title(f'线路{line} · {hour}:00 · 不同天气预测对比', fontsize=28, fontweight='bold')
        self.canvas.ax.tick_params(labelsize=24)
        self.canvas.ax.grid(axis='y', alpha=0.3)
        self.canvas.fig.tight_layout(pad=3)
        self.canvas.draw()
        
        # 表格
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(['天气类型', '预测乘车量', '建议车辆数', '较舒适变化'])
        self.result_table.setRowCount(len(weathers))
        for i, (w, pred, bus) in enumerate(zip(weathers, preds, buses)):
            if i == 0:
                change = '---'
            else:
                pct = (pred - comfort_pred) / comfort_pred * 100
                change = f'{pct:+.1f}%'
            items = [
                QTableWidgetItem(w),
                QTableWidgetItem(f'{pred:.0f} 人'),
                QTableWidgetItem(f'{bus} 辆'),
                QTableWidgetItem(change),
            ]
            for j, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                if i == 0:
                    item.setBackground(QColor('#F0FFF0'))
                self.result_table.setItem(i, j, item)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # ============================================================
    # Tab 3: 多线路对比
    # ============================================================
    def update_multiline(self):
        hour = int(self.ml_hour_combo.currentText().replace(':00', ''))
        weather = self.ml_weather_combo.currentText()
        day_type = self.ml_day_combo.currentText()
        
        temp, tp = WEATHER_PRESETS[weather]
        if day_type == '工作日':
            is_holiday, weekday = 0, 2
        elif day_type == '周末':
            is_holiday, weekday = 0, 6
        else:
            is_holiday, weekday = 1, 6
        
        # 预测所有线路
        lines = self.top_lines[:10]
        preds = []
        buses = []
        for line in lines:
            X = self.make_input(hour, line, temp, tp, is_holiday, weekday, 11)
            pred = self.predict(X)[0]
            bus = max(1, int(np.ceil(pred / PASSENGERS_PER_BUS)))
            preds.append(pred)
            buses.append(bus)
        
        # 绘图
        self.canvas.ax.clear()
        colors = []
        for i, bus in enumerate(buses):
            if bus >= 7:
                colors.append('#FF6B6B')
            elif bus >= 4:
                colors.append('#4ECDC4')
            else:
                colors.append('#45B7D1')
        
        line_labels = [f'线路{l}' for l in lines]
        bars = self.canvas.ax.barh(range(len(lines)), preds, color=colors, alpha=0.85, edgecolor='white')
        
        for i, (pred, bus) in enumerate(zip(preds, buses)):
            self.canvas.ax.text(pred + max(preds) * 0.02, i, f'{pred:.0f}人 ({bus}辆)',
                               va='center', fontsize=24, fontweight='bold')
        
        self.canvas.ax.set_yticks(range(len(lines)))
        self.canvas.ax.set_yticklabels(line_labels, fontsize=24)
        self.canvas.ax.set_xlabel('预测乘车量（人）', fontsize=24)
        self.canvas.ax.set_title(f'{hour}:00 · {day_type} · {weather} · 各线路对比', fontsize=28, fontweight='bold')
        self.canvas.ax.invert_yaxis()
        self.canvas.ax.tick_params(labelsize=24)
        self.canvas.ax.grid(axis='x', alpha=0.3)
        self.canvas.fig.tight_layout(pad=3)
        self.canvas.draw()
        
        # 表格
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(['线路', '预测乘车量', '建议车辆数', '建议发车间隔', '运营建议'])
        self.result_table.setRowCount(len(lines))
        for i, (line, pred, bus) in enumerate(zip(lines, preds, buses)):
            interval = 60 / bus
            if bus >= 7:
                advice = '重点线路，加密发车'
            elif bus >= 4:
                advice = '常规线路，标准发车'
            else:
                advice = '低客流，拉大间隔'
            
            items = [
                QTableWidgetItem(f'{line}'),
                QTableWidgetItem(f'{pred:.0f} 人'),
                QTableWidgetItem(f'{bus} 辆'),
                QTableWidgetItem(f'{interval:.1f} 分钟'),
                QTableWidgetItem(advice),
            ]
            for j, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                if bus >= 7:
                    item.setBackground(QColor('#FFF0F0'))
                elif bus <= 2:
                    item.setBackground(QColor('#F0F8FF'))
                self.result_table.setItem(i, j, item)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 14))
    app.setStyleSheet(STYLE_SHEET)
    window = PredictionApp()
    sys.exit(app.exec_())