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

# 關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================================================================
# 🌐 雲端中文字型純淨相容設定
# ==================================================================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 🔑 氣象署 API 授權碼
CWA_API_KEY = "CWA-5D29B5B2-7FEA-4B0A-A482-92BC6D8FCFBC"

# ==================================================================
# API 與推算引擎
# ==================================================================
@st.cache_data(ttl=3600)
def fetch_real_wind_data(api_key, target_date_str):
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
                for h in range(hr, min(hr+3, 25)):
                    dirs[h] = deg; speeds[h] = spd; waves[h] = spd * 0.2  
        if has_data: return hours, dirs, speeds, waves
    except Exception: pass
    return None

def calculate_daily_tide_profile(year, month, day):
    lunar_date = LunarDate.fromSolarDate(year, month, day)
    d_lunar = lunar_date.day
    k_today = abs(math.sin((2 * math.pi * (d_lunar - 1.5)) / 29.53))
    ht1 = (d_lunar if d_lunar <= 15 else d_lunar - 15) - 1 * 0.8 + 0.5
    return d_lunar, k_today, ht1, np.linspace(0, 24, 240), 100 * (0.4 + 0.6 * k_today) * np.cos(2 * math.pi * (np.linspace(0, 24, 240) - ht1) / 12.42)

def simulate_daily_wind_forecast(target_date):
    has_front = (target_date.timetuple().tm_yday % 7 == 5) 
    hours, dirs, speeds, waves = [], [], [], []
    for h in range(25):
        hours.append(h)
        if has_front and h >= 13: dirs.append(45); speeds.append(12.5); waves.append(1.8)
        else: dirs.append(225 if 5 <= target_date.month <= 8 else 45); speeds.append(4.5); waves.append(0.3)
    return np.array(hours), np.array(dirs), np.array(speeds), np.array(waves)

def calculate_qimen_tianfu(target_date, target_hour):
    zhi_idx = ((target_hour + 1) // 2) % 12
    zhi_names = ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]
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
    ax.set_title(f"[Chart 5] Qi Men Dun Jia (Time: {target_hour:02d}:00 {shi_chen})", weight='bold', color='#4b0082')
    ax.axis('off')
    for i in range(4): ax.axhline(i, color='#808080', lw=2); ax.axvline(i, color='#808080', lw=2)
    palaces = {(0, 2): ("Xun(SE)", "Tian Fu", "Du Men", "9 Di"),  (1, 2): ("Li(S)", "Tian Ying", "Jing Men", "9 Tian"),   (2, 2): ("Kun(SW)", "Tian Rui", "Si Men", "Fu"),
               (0, 1): ("Zhen(E)", "Tian Chong", "Shang", "Xuan"), (1, 1): ("Center", "", "", ""),                  (2, 1): ("Dui(W)", "Tian Zhu", "Jing Men", "She"),
               (0, 0): ("Gen(NE)", "Tian Ren", "Sheng", "Bai"),    (1, 0): ("Kan(N)", "Tian Peng", "Xiu Men", "He"),     (2, 0): ("Qian(NW)", "Tian Xin", "Kai Men", "Yin")}
    for (x, y), (gua, star, door, god) in palaces.items():
        cx, cy = x + 0.5, y + 0.5
        if x == wind_x and y == wind_y and star != "":
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='#e6e6fa', zorder=1)) 
            ax.text(cx, cy+0.25, "*Tian Fu (Wind)", ha='center', va='center', color='purple', weight='bold', fontsize=11)
        elif star:
            ax.text(cx, cy+0.25, star if star != "Tian Fu" else "Xun", ha='center', va='center', color='black', fontsize=10)
        ax.text(cx, cy, gua, ha='center', va='center', color='#8b0000', weight='bold', fontsize=12)
        if door:
            ax.text(cx, cy-0.25, f"{god}/{door}", ha='center', va='center', color='green', fontsize=10)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect('equal')

# ==================================================================
# Streamlit 主網頁
# ==================================================================
st.set_page_config(page_title="Guishan Island Dashboard", layout="wide")
st.title("⚓ 龜山島五星動態決策儀表板")

st.sidebar.header("⚙️ Settings")
default_date = datetime.date.today() + datetime.timedelta(days=1)
target_date = st.sidebar.date_input("Target Date", default_date)
alert_tide = st.sidebar.number_input("Alert Tide (cm)", value=-60)
alert_wind = st.sidebar.number_input("Alert Wind (m/s)", value=10.8)
alert_wave = st.sidebar.number_input("Alert Wave (m)", value=1.5)

target_date_str = target_date.strftime('%Y-%m-%d')
target_dt = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')

st.header(f"🎯 Tactical Analysis: {target_date_str}")
d_lunar, k_today, ht1, tide_hours, tide_heights = calculate_daily_tide_profile(target_dt.year, target_dt.month, target_dt.day)
api_data = fetch_real_wind_data(CWA_API_KEY, target_date_str)

if api_data: 
    f_hours, f_dirs, f_speeds, f_waves = api_data
    st.success("🟢 成功連接中央氣象署即時 API")
else: 
    f_hours, f_dirs, f_speeds, f_waves = simulate_daily_wind_forecast(target_dt)
    st.warning("🟡 目前使用歷史模型模擬數據")

target_qimen_hour = 12 
shift_idx = np.where(np.diff(f_dirs) != 0)[0]
if len(shift_idx) > 0: target_qimen_hour = int(f_hours[shift_idx[0] + 1]) 

fig2 = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.5, 1.5], wspace=0.25, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0]); ax5 = fig2.add_subplot(gs[1, 0])
ax2 = fig2.add_subplot(gs[0, 1]); ax3 = fig2.add_subplot(gs[0, 2]); ax4 = fig2.add_subplot(gs[1, 1:])

# 圖一羅盤
ax1.scatter(0, 0, color='#1f77b4', s=400, zorder=5)
ax1.text(0, -0.2, "Earth", ha='center', weight='bold', color='#1f77b4')
r_c = 1.6; theta = np.linspace(0, 2*math.pi, 150); ax1.plot(r_c*np.cos(theta), r_c*np.sin(theta), color='gray')
bagua = {0:"Zhen(E)", 45:"Xun(SE)", 90:"Li(S)", 135:"Kun(SW)", 180:"Dui(W)", 225:"Qian(NW)", 270:"Kan(N)", 315:"Gen(NE)"}
for angle, label in bagua.items():
    rad = math.radians(angle)
    ax1.text(r_c*math.cos(rad)*1.2, r_c*math.sin(rad)*1.2, label, ha='center', va='center', weight='bold')
ax1.set_title("[Chart 1] Compass & Lunar Phase", weight='bold'); ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5); ax1.set_aspect('equal')

# 圖五奇門
draw_qimen_chart(ax5, target_date, target_qimen_hour)

# 圖二潮汐
ax2.plot(tide_hours, tide_heights, color='#008080', lw=2.5)
ax2.axhline(y=alert_tide, color='red', ls='--')
ax2.set_title("[Chart 2] Tide Level (09:00-16:00)", weight='bold'); ax2.set_xlim(9, 16); ax2.set_ylim(-150, 150); ax2.grid(True, ls=':')

# 圖三風速
ax3.plot(f_hours, f_speeds, color='#1f77b4', lw=2.5)
ax3.axhline(y=alert_wind, color='red', ls='--')
ax3.set_title("[Chart 3] Wind Speed (09:00-16:00)", weight='bold'); ax3.set_xlim(9, 16); ax3.set_ylim(0, 16); ax3.grid(True, ls=':')

# 圖四波高
ax4.plot(f_hours, f_waves, color='#2ca02c', lw=3)
ax4.axhline(y=alert_wave, color='red', ls='--')
ax4.set_title("[Chart 4] Wave Height (09:00-16:00)", weight='bold'); ax4.set_xlim(9, 16); ax4.set_ylim(0, 2.5); ax4.grid(True, ls=':')

st.pyplot(fig2)

# 文字預報
st.header("📋 戰略參謀總結與一週預報")
t_mask_tmr = (tide_hours >= 9.0) & (tide_hours <= 16.0)
min_tide_tmr = np.min(tide_heights[t_mask_tmr])
overall_status = "🟢 【海象平穩】適宜開航"
if min_tide_tmr <= alert_tide: overall_status = "🟡 【乾潮警戒】南岸面臨觸底危機，建議避開中午靠泊！"
if api_data:
    mask_tmr = (np.arange(25) >= 9) & (np.arange(25) <= 16)
    if np.max(f_speeds[mask_tmr]) >= alert_wind or np.max(f_waves[mask_tmr]) >= alert_wave:
        overall_status = "🔴 【封島警戒】強風大浪超標，強烈建議停航！"

st.info(f"**長官決策建議：** {overall_status}")

forecast_data = []
for i in range(7):
    curr_dt = target_dt + datetime.timedelta(days=i)
    _, _, _, t_hrs, t_hts = calculate_daily_tide_profile(curr_dt.year, curr_dt.month, curr_dt.day)
    min_tide = np.min(t_hts[(t_hrs >= 9.0) & (t_hrs <= 16.0)])
    tide_msg = f"{min_tide:5.1f} cm" + (" (⚠️ 觸底)" if min_tide <= alert_tide else "")
    
    api_d = fetch_real_wind_data(CWA_API_KEY, curr_dt.strftime('%Y-%m-%d'))
    if api_d: _, _, spds, waves = api_d; src = "API"
    else: _, _, spds, waves = simulate_daily_wind_forecast(curr_dt); src = "Model"
    w_mask = (np.arange(25) >= 9) & (np.arange(25) <= 16)
    forecast_data.append({"日期": curr_dt.strftime('%m/%d (%a)'), "來源": src, "最低潮位": tide_msg, "最大風速": f"{np.max(spds[w_mask]):.1f} m/s", "最大浪高": f"{np.max(waves[w_mask]):.1f} m"})

st.table(pd.DataFrame(forecast_data))
