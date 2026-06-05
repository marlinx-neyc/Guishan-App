import streamlit as st
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from lunardate import LunarDate
import requests
import urllib3
import os
import urllib.request
import matplotlib.font_manager as fm

# 關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ==================================================================
# 🌐 雲端中文字型自動載入模組 (解決 Linux 伺服器亂碼問題)
# ==================================================================
@st.cache_resource
def load_font():
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/TTF/NotoSansCJKtc-VF.ttf"
    font_path = "NotoSansCJKtc-VF.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    plt.rcParams['axes.unicode_minus'] = False


load_font()

# 🔑 氣象署 API 授權碼
CWA_API_KEY = "CWA-5D29B5B2-7FEA-4B0A-A482-92BC6D8FCFBC"


# ==================================================================
# API 與推算引擎
# ==================================================================
@st.cache_data(ttl=3600)  # 快取 1 小時，避免頻繁打 API
def fetch_real_wind_data(api_key, target_date_str):
    target_date_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')
    try:
        url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-001?Authorization={api_key}&locationName=頭城鎮"
        res = requests.get(url, verify=False, timeout=8)
        if res.status_code != 200: return None
        data = res.json()
        weathers = data['records']['locations'][0]['location'][0]['weatherElement']
        wd_data = next(e for e in weathers if e['elementName'] == 'WD')['time']
        ws_data = next(e for e in weathers if e['elementName'] == 'WS')['time']

        hours, dirs, speeds, waves = np.linspace(0, 24, 25), np.zeros(25), np.zeros(25), np.zeros(25)
        has_data = False
        dir_map = {"北": 360, "東北": 45, "東": 90, "東南": 135, "南": 180, "西南": 225, "西": 270, "西北": 315}

        for i in range(len(wd_data)):
            st_time = wd_data[i]['startTime']
            if st_time.startswith(target_date_str):
                has_data = True
                hr = datetime.datetime.strptime(st_time, '%Y-%m-%d %H:%M:%S').hour
                deg = dir_map.get(wd_data[i]['elementValue'][0]['value'], 0)
                spd = float(ws_data[i]['elementValue'][0]['value'])
                for h in range(hr, min(hr + 3, 25)):
                    dirs[h] = deg;
                    speeds[h] = spd;
                    waves[h] = spd * 0.2
        if has_data: return hours, dirs, speeds, waves
    except Exception:
        pass
    return None


def calculate_daily_tide_profile(year, month, day):
    lunar_date = LunarDate.fromSolarDate(year, month, day)
    d_lunar = lunar_date.day
    k_today = abs(math.sin((2 * math.pi * (d_lunar - 1.5)) / 29.53))
    ht1 = (d_lunar if d_lunar <= 15 else d_lunar - 15) - 1 * 0.8 + 0.5
    return d_lunar, k_today, ht1, np.linspace(0, 24, 240), 100 * (0.4 + 0.6 * k_today) * np.cos(
        2 * math.pi * (np.linspace(0, 24, 240) - ht1) / 12.42)


def simulate_daily_wind_forecast(target_date):
    has_front = (target_date.timetuple().tm_yday % 7 == 5)
    hours, dirs, speeds, waves = [], [], [], []
    for h in range(25):
        hours.append(h)
        if has_front and h >= 13:
            dirs.append(45); speeds.append(12.5); waves.append(1.8)
        else:
            dirs.append(225 if 5 <= target_date.month <= 8 else 45); speeds.append(4.5); waves.append(0.3)
    return np.array(hours), np.array(dirs), np.array(speeds), np.array(waves)


# ==================================================================
# 奇門遁甲核心引擎
# ==================================================================
def calculate_qimen_tianfu(target_date, target_hour):
    zhi_idx = ((target_hour + 1) // 2) % 12
    zhi_names = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    shi_chen = zhi_names[zhi_idx]
    is_yang_dun = 80 <= target_date.timetuple().tm_yday <= 266
    delta_days = (target_date - datetime.date(2026, 1, 1)).days
    hour_gan = (((4 + delta_days) % 10 % 5) * 2 + zhi_idx) % 10
    base_palace = 4
    shift = (hour_gan + zhi_idx) % 9 if is_yang_dun else -(hour_gan + zhi_idx) % 9
    current_palace = (base_palace + shift - 1) % 9 + 1
    luoshu_coords = {4: (0, 2), 9: (1, 2), 2: (2, 2), 3: (0, 1), 5: (1, 1), 7: (2, 1), 8: (0, 0), 1: (1, 0), 6: (2, 0)}
    return luoshu_coords.get(current_palace, (0, 2)), shi_chen


def draw_qimen_chart(ax, target_date, target_hour):
    (wind_x, wind_y), shi_chen = calculate_qimen_tianfu(target_date, target_hour)
    ax.set_title(f"[圖五] 時空奇門盤 (標定時辰: {target_hour:02d}:00 {shi_chen}時)", weight='bold', color='#4b0082')
    ax.axis('off')
    for i in range(4): ax.axhline(i, color='#808080', lw=2); ax.axvline(i, color='#808080', lw=2)
    palaces = {(0, 2): ("巽(東南)", "天輔", "杜門", "九地"), (1, 2): ("離(南)", "天英", "景門", "九天"),
               (2, 2): ("坤(西南)", "天芮", "死門", "值符"),
               (0, 1): ("震(東)", "天沖", "傷門", "玄武"), (1, 1): ("中宮", "", "", ""),
               (2, 1): ("兌(西)", "天柱", "驚門", "螣蛇"),
               (0, 0): ("艮(東北)", "天任", "生門", "白虎"), (1, 0): ("坎(北)", "天蓬", "休門", "六合"),
               (2, 0): ("乾(西北)", "天心", "開門", "太陰")}
    for (x, y), (gua, star, door, god) in palaces.items():
        cx, cy = x + 0.5, y + 0.5
        if x == wind_x and y == wind_y and star != "":
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='#e6e6fa', zorder=1))
            ax.text(cx, cy + 0.25, f"★天輔(主風)", ha='center', va='center', color='purple', weight='bold', fontsize=11)
        elif star:
            ax.text(cx, cy + 0.25, star if star != "天輔" else "巽(本位)", ha='center', va='center', color='black',
                    fontsize=10)
        ax.text(cx, cy, gua, ha='center', va='center', color='#8b0000', weight='bold', fontsize=12)
        if door:
            color_door = 'red' if door in ['開門', '休門', '生門', '景門'] else 'green'
            ax.text(cx, cy - 0.25, f"{god} / {door}", ha='center', va='center', color=color_door, fontsize=10)
    ax.set_xlim(0, 3);
    ax.set_ylim(0, 3);
    ax.set_aspect('equal')


# ==================================================================
# Streamlit 網頁主程式
# ==================================================================
st.set_page_config(page_title="龜山島動態決策儀表板", layout="wide")

st.title("⚓ 龜山島五星動態決策儀表板")
st.markdown("結合氣象署即時 API、農曆潮汐推算與奇門遁甲預測的決策輔助系統。")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 參數設定")
default_date = datetime.date.today() + datetime.timedelta(days=1)
target_date = st.sidebar.date_input("選擇推演日期", default_date)
alert_tide = st.sidebar.number_input("觸底警戒水位 (cm)", value=-60)
alert_wind = st.sidebar.number_input("封島警戒風速 (m/s)", value=10.8)
alert_wave = st.sidebar.number_input("封島警戒浪高 (m)", value=1.5)

target_date_str = target_date.strftime('%Y-%m-%d')
target_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')

# ==================================================================
# 單日戰術儀表板渲染
# ==================================================================
st.header(f"🎯 單日戰術解析：{target_date_str}")

d_lunar, k_today, ht1, tide_hours, tide_heights = calculate_daily_tide_profile(target_dt.year, target_dt.month,
                                                                               target_dt.day)
api_data = fetch_real_wind_data(CWA_API_KEY, target_date_str)

if api_data:
    f_hours, f_dirs, f_speeds, f_waves = api_data
    st.success("🟢 成功連接氣象署即時 API 數據")
else:
    f_hours, f_dirs, f_speeds, f_waves = simulate_daily_wind_forecast(target_dt)
    st.warning("🟡 無法取得 API 數據 (可能超過預報範圍)，已切換為模型推算")

target_qimen_hour = 12
shift_idx = np.where(np.diff(f_dirs) != 0)[0]
if len(shift_idx) > 0:
    target_qimen_hour = int(f_hours[shift_idx[0] + 1])

fig2 = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.5, 1.5], wspace=0.25, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0]);
ax5 = fig2.add_subplot(gs[1, 0])
ax2 = fig2.add_subplot(gs[0, 1]);
ax3 = fig2.add_subplot(gs[0, 2]);
ax4 = fig2.add_subplot(gs[1, 1:])

# 圖一羅盤與月相
ax1.scatter(0, 0, color='#1f77b4', s=400, zorder=5)
ax1.text(0, -0.2, "地球", ha='center', weight='bold', color='#1f77b4')
r_c = 1.6;
theta = np.linspace(0, 2 * math.pi, 150)
ax1.plot(r_c * np.cos(theta), r_c * np.sin(theta), color='#9c9c9c', ls='-')
bagua = {0: "震", 45: "巽", 90: "離", 135: "坤", 180: "兌", 225: "乾", 270: "坎", 315: "艮"}
for angle, label in bagua.items():
    rad = math.radians(angle)
    ax1.plot([0, r_c * math.cos(rad)], [0, r_c * math.sin(rad)], color='gray', ls='--', alpha=0.5)
    ax1.text(r_c * math.cos(rad) * 1.18, r_c * math.sin(rad) * 1.18, label, ha='center', va='center', weight='bold')
compass_pts = {0: "東(E)", 90: "南(S)", 180: "西(W)", 270: "北(N)"}
for angle, label in compass_pts.items():
    rad = math.radians(angle)
    ax1.text(r_c * math.cos(rad) * 1.45, r_c * math.sin(rad) * 1.45, label, ha='center', va='center', weight='bold',
             color='darkred', fontsize=12)

lunar_rad = (d_lunar - 1) * (2 * math.pi / 29.53)
ax1.scatter(2.1, 0.0, color='red', s=600, zorder=5)
ax1.scatter(1.0 * math.cos(lunar_rad), 1.0 * math.sin(lunar_rad), color='#e0e0e0', s=150, zorder=5, ec='black')
phase_rad = (d_lunar / 29.53) * 2 * math.pi
cx, cy, R_moon = -2.1, 2.1, 0.45
ax1.add_patch(plt.Circle((cx, cy), R_moon, color='#333333', zorder=10))
y_pts = np.linspace(-R_moon, R_moon, 100);
x_out_r = np.sqrt(R_moon ** 2 - y_pts ** 2)
if phase_rad <= math.pi:
    ax1.fill_betweenx(y_pts + cy, cx + np.cos(phase_rad) * x_out_r, cx + x_out_r, color='#FFD700', zorder=11)
else:
    ax1.fill_betweenx(y_pts + cy, cx - x_out_r, cx - np.cos(phase_rad) * x_out_r, color='#FFD700', zorder=11)
ax1.add_patch(plt.Circle((cx, cy), R_moon, color='#9c9c9c', fill=False, lw=1.5, zorder=12))
ax1.text(cx, cy - R_moon - 0.15, f"當前月相\n(農曆初{int(d_lunar)})", ha='center', va='top', weight='bold',
         color='#333333', bbox=dict(fc='#f0f0f0', alpha=0.8, boxstyle='round,pad=0.2'))
ax1.set_title("[圖一] 納甲羅盤與即時月相", weight='bold');
ax1.set_xlim(-2.8, 3.0);
ax1.set_ylim(-2.8, 2.8);
ax1.set_aspect('equal')

draw_qimen_chart(ax5, target_date, target_qimen_hour)

# 潮汐、風向、海浪圖
ax2.plot(tide_hours, tide_heights, color='#008080', lw=2.5)
ax2.axhline(y=alert_tide, color='red', ls='--', lw=2)
ax2.fill_between(tide_hours, tide_heights, alert_tide, where=(tide_heights < alert_tide), color='red', alpha=0.3)
for ht in [ht1, (ht1 + 12.42) % 24]:
    if 9 <= ht <= 16:
        idx = np.abs(tide_hours - ht).argmin()
        ax2.scatter(ht, tide_heights[idx], color='red', s=40, zorder=5)
        ax2.text(ht, tide_heights[idx] + 15, f"{int(ht):02d}:{int((ht % 1) * 60):02d}", ha='center', color='red',
                 weight='bold')
ax2.set_title("[圖二] 南岸觸底警戒區 (09:00-16:00)", weight='bold');
ax2.set_xlim(9, 16);
ax2.set_ylim(-150, 150);
ax2.grid(True, ls=':', alpha=0.5)

ax3.plot(f_hours, f_speeds, color='#1f77b4', lw=2.5)
ax3.axhline(y=alert_wind, color='red', ls='--', lw=2)
for i in range(9, 17):
    if i % 2 == 0:
        u, v = np.cos(math.radians(270 - f_dirs[i])), np.sin(math.radians(270 - f_dirs[i]))
        ax3.quiver(f_hours[i], f_speeds[i] + 1, u, v, color='purple', scale=20)
ax3.fill_between(f_hours, f_speeds, alert_wind, where=(f_speeds >= alert_wind), color='red', alpha=0.2)
if len(shift_idx) > 0:
    shift_time = f_hours[shift_idx[0] + 1]
    if 9 <= shift_time <= 16:
        ax3.axvline(x=shift_time, color='orange', ls='-.', lw=2)
        ax3.text(shift_time - 0.2, 12, f"注意 {int(shift_time):02d}:00\n轉向", color='#8b0000', weight='bold',
                 ha='right')
ax3.set_title("[圖三] 實時風向切變與風速 (09:00-16:00)", weight='bold');
ax3.set_xlim(9, 16);
ax3.set_ylim(0, 16);
ax3.grid(True, ls=':', alpha=0.5)

ax4.plot(f_hours, f_waves, color='#2ca02c', lw=3)
ax4.axhline(y=0.5, color='orange', ls='--', lw=2);
ax4.axhline(y=alert_wave, color='red', ls='-', lw=2)
ax4.fill_between(f_hours, f_waves, 0.5, where=(f_waves >= 0.5), color='orange', alpha=0.2)
ax4.fill_between(f_hours, f_waves, alert_wave, where=(f_waves >= alert_wave), color='red', alpha=0.4)
if len(shift_idx) > 0 and 'shift_time' in locals() and shift_time is not None:
    if 9 <= shift_time <= 16:
        ax4.axvline(x=shift_time, color='orange', ls='-.', lw=2)
        ax4.text(shift_time + 0.2, 1.6, "北岸癱瘓", color='#8b0000', weight='bold')
ax4.set_title("[圖四] 實時波高與北岸警戒 (09:00-16:00)", weight='bold');
ax4.set_xlim(9, 16);
ax4.set_ylim(0, 2.5);
ax4.grid(True, ls=':', alpha=0.5)

# 將圖表渲染到網頁
st.pyplot(fig2)

# ==================================================================
# 一週水文預報 (文字報告區塊)
# ==================================================================
st.header("📋 戰略參謀總結與一週預報")
t_mask_tmr = (tide_hours >= 9.0) & (tide_hours <= 16.0)
min_tide_tmr = np.min(tide_heights[t_mask_tmr])
overall_status = "🟢 【海象平穩】適宜開航，留意例行安檢即可。"
if min_tide_tmr <= alert_tide: overall_status = "🟡 【乾潮警戒】南岸將面臨淺水危機，請避開中午時段靠泊！"
if api_data:
    mask_tmr = (np.arange(25) >= 9) & (np.arange(25) <= 16)
    if np.max(f_speeds[mask_tmr]) >= alert_wind or np.max(f_waves[mask_tmr]) >= alert_wave:
        overall_status = "🔴 【封島警戒】氣象署預測強風/大浪超標，強烈建議全日停航！"

st.info(f"**明日長官決策建議：** {overall_status}")

forecast_data = []
for i in range(7):
    curr_dt = target_dt + datetime.timedelta(days=i)
    curr_str = curr_dt.strftime('%m/%d (%a)')
    _, _, _, t_hrs, t_hts = calculate_daily_tide_profile(curr_dt.year, curr_dt.month, curr_dt.day)
    min_tide = np.min(t_hts[(t_hrs >= 9.0) & (t_hrs <= 16.0)])
    tide_msg = f"{min_tide:5.1f} cm" + (" (⚠️ 觸底)" if min_tide <= alert_tide else "")

    api_d = fetch_real_wind_data(CWA_API_KEY, curr_dt.strftime('%Y-%m-%d'))
    if api_d:
        _, _, spds, waves = api_d; src = "API"
    else:
        _, _, spds, waves = simulate_daily_wind_forecast(curr_dt); src = "模型"

    w_mask = (np.arange(25) >= 9) & (np.arange(25) <= 16)
    max_wind = np.max(spds[w_mask])
    wind_msg = f"{max_wind:4.1f} m/s" + (" (❌ 封島)" if max_wind >= alert_wind else "")
    forecast_data.append({"日期": curr_str, "資料來源": src, "最低潮位": tide_msg, "最大風速": wind_msg,
                          "最大浪高": f"{np.max(waves[w_mask]):3.1f} m"})

st.table(pd.DataFrame(forecast_data))

# ==================================================================
# 宏觀戰略排查與 CSV 匯出
# ==================================================================
st.header("🗺️ 年度宏觀戰略排查 (3-10月)")
with st.spinner('計算宏觀趨勢中...'):
    start_date, end_date = datetime.date(target_date.year, 3, 1), datetime.date(target_date.year, 10, 31)
    macro_tide_dates, macro_tide_vals = [], []
    macro_wind_dates, macro_wind_vals = [], []
    alert_list = []

    curr_date = start_date
    while curr_date <= end_date:
        _, _, _, t_hrs, t_hts = calculate_daily_tide_profile(curr_date.year, curr_date.month, curr_date.day)
        w_hrs, w_dirs, w_spds, _ = simulate_daily_wind_forecast(curr_date)
        date_str_md = curr_date.strftime('%m/%d')

        t_mask = (t_hrs >= 9.0) & (t_hrs <= 16.0)
        min_tide = np.min(t_hts[t_mask])
        min_time = t_hrs[t_mask][np.argmin(t_hts[t_mask])]
        if min_tide <= alert_tide:
            macro_tide_dates.append(curr_date);
            macro_tide_vals.append(min_tide)
            alert_list.append(
                {"類別": "大乾潮", "日期": date_str_md, "時間": f"{int(min_time):02d}:{int((min_time % 1) * 60):02d}",
                 "數值": f"{min_tide:.1f} cm"})

        w_mask = (w_hrs >= 9.0) & (w_hrs <= 16.0)
        if np.max(w_spds[w_mask]) >= alert_wind:
            shift_idx = np.where(np.diff(w_dirs[w_mask]) != 0)[0]
            if len(shift_idx) > 0:
                shift_t = w_hrs[w_mask][shift_idx[0] + 1]
                macro_wind_dates.append(curr_date);
                macro_wind_vals.append(np.max(w_spds[w_mask]))
                alert_list.append({"類別": "轉強風", "日期": date_str_md, "時間": f"{int(shift_t):02d}:00",
                                   "數值": f"{np.max(w_spds[w_mask]):.1f} m/s"})
        curr_date += datetime.timedelta(days=1)

    fig1, axs1 = plt.subplots(2, 1, figsize=(16, 8))
    ax_m_tide, ax_m_wind = axs1[0], axs1[1]

    for ax in [ax_m_tide, ax_m_wind]:
        ax.axvspan(datetime.date(target_date.year, 3, 1), datetime.date(target_date.year, 5, 31), color='#eaffea',
                   alpha=0.5)
        ax.axvspan(datetime.date(target_date.year, 6, 1), datetime.date(target_date.year, 8, 31), color='#ffeaea',
                   alpha=0.5)
        ax.axvspan(datetime.date(target_date.year, 9, 1), datetime.date(target_date.year, 10, 31), color='#fff9e6',
                   alpha=0.5)

    ax_m_tide.axhline(alert_tide, color='red', ls='--', lw=2)
    if macro_tide_dates:
        ax_m_tide.vlines(macro_tide_dates, alert_tide, macro_tide_vals, color='#d62728', lw=2, alpha=0.7)
        ax_m_tide.scatter(macro_tide_dates, macro_tide_vals, color='#d62728', s=30, zorder=5)
    ax_m_tide.xaxis.set_major_locator(mdates.MonthLocator());
    ax_m_tide.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    ax_m_tide.set_title("【戰略】3-10月大乾潮分佈", weight='bold');
    ax_m_tide.set_ylim(-125, 0);
    ax_m_tide.grid(True, ls=':', alpha=0.5)

    ax_m_wind.axhline(alert_wind, color='red', ls='--', lw=2)
    if macro_wind_dates:
        ax_m_wind.vlines(macro_wind_dates, alert_wind, macro_wind_vals, color='#ff7f0e', lw=2, alpha=0.7)
        ax_m_wind.scatter(macro_wind_dates, macro_wind_vals, color='#ff7f0e', s=40, zorder=5)
    ax_m_wind.xaxis.set_major_locator(mdates.MonthLocator());
    ax_m_wind.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    ax_m_wind.set_title("【戰略】3-10月突變轉強風趨勢", weight='bold');
    ax_m_wind.set_ylim(0, 16);
    ax_m_wind.grid(True, ls=':', alpha=0.5)

    fig1.tight_layout(pad=3.0)
    st.pyplot(fig1)

    # 顯示並提供下載 CSV
    if alert_list:
        df_alerts = pd.DataFrame(alert_list)
        csv_data = df_alerts.to_csv(index=False).encode('utf-8-sig')

        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(label="📥 下載戰略預警清單 (CSV)", data=csv_data,
                               file_name=f"龜山島戰略預警清單_{target_date.year}.csv", mime="text/csv")
        with col2:
            with st.expander("👀 點擊展開線上檢視預警清單"):
                st.dataframe(df_alerts, use_container_width=True)

st.caption("Designed for 龜山島戰術指揮中心 | Powered by CWA & Lunar Math Engine")