import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# --- CORE IMPORTS ---
try:
    from core.data_manager import DataManager
    from core.ml_engine import ApexML
    from core.strategy_engine import RaceSimulator, StrategyConfig
    from core.analytics import CornerAnalyst, HistoricalVault
    from core.insight_engine import InsightEngine
    from core.report_gen import ReportGenerator
    from core.results_manager import ResultsManager
    from core.mcda_engine_v2 import F1DecisionEngine, CircuitParams, CarParams, EnvParams, TireParams
except ImportError as e:
    st.error(f"⚠️ CORE MODULE MISSING: {e}")
    st.stop()

# ------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(layout="wide", page_title="ApexBrain PitWall", page_icon="🏎️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        background-color: #0e1117; 
        color: #e2e8f0; 
    }
    
    /* CARDS */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: #1e293b; 
        border: 1px solid #334155; 
        border-radius: 8px; 
        padding: 15px;
    }
    
    /* CUSTOM METRICS */
    .metric-container {
        text-align: center;
        padding: 10px;
        background: #0f172a;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .metric-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: 800; color: #f8fafc; }
    
    /* TIMING TOWER */
    .tower-row {
        display: flex; justify-content: space-between;
        padding: 8px; border-bottom: 1px solid #334155;
        font-family: 'Courier New', monospace; font-weight: bold; font-size: 14px;
    }
    .pos-box { width: 30px; text-align: center; background: #334155; color: white; border-radius: 4px; margin-right: 10px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 2. STATE INITIALIZATION
# ------------------------------------------------------
if 'dm' not in st.session_state: st.session_state.dm = DataManager()
if 'ml' not in st.session_state: st.session_state.ml = ApexML()
if 'analyst' not in st.session_state: st.session_state.analyst = CornerAnalyst()
if 'insight' not in st.session_state: st.session_state.insight = InsightEngine()
if 'res' not in st.session_state: st.session_state.res = ResultsManager()
if 'user_stints' not in st.session_state: st.session_state.user_stints = []
if 'saved_strategies' not in st.session_state: st.session_state.saved_strategies = []

# ------------------------------------------------------
# 3. SIDEBAR (CONNECTION)
# ------------------------------------------------------
with st.sidebar:
    st.title("🏎️ ApexBrain")
    st.caption("PITWALL EDITION v24.0")
    st.markdown("---")
    
    s_year = st.selectbox("Season", [2025, 2024, 2023], index=1)
    s_gp = st.selectbox("Grand Prix", ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Miami", "Monaco", "Spa", "Italy"], index=0)
    s_session = st.selectbox("Session", ["Race", "Qualifying"], index=0)
    
    if st.button("📡 CONNECT FEED", type="primary", use_container_width=True):
        with st.spinner("Syncing Telemetry..."):
            success, msg = st.session_state.dm.load_session(s_year, s_gp, s_session)
            if success: st.toast("Connected", icon="✅")
            else: st.error(msg)
            
    if st.session_state.dm.session:
        st.success("🟢 LIVE DATA STREAM")

# ------------------------------------------------------
# 4. MAIN TABS
# ------------------------------------------------------
tabs = st.tabs(["🧠 STRATEGY", "📡 LIVE TRACK", "⚔️ BATTLE", "📉 STINTS", "🏎️ CONCEPT", "🔮 RACE STRAT", "📊 TELEMETRY", "🧪 ML LAB", "🚀 MATRIX"])

# ==============================================================================
# TAB 1: DECISION SUPPORT (PIT WALL - ORGANIZED)
# ==============================================================================
with tabs[0]:
    from core.mcda_engine_v2 import F1DecisionEngine, CircuitParams, CarParams, EnvParams, TireParams
    
    # ------------------------------------------------------------------
    # 1. TOP: MISSION CONTROL (RE-ORGANIZED LAYOUT)
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 🎛️ Mission Control")
        
        # Create 3 distinct columns with larger gap for visual separation
        col_race, col_cond, col_ai = st.columns([1, 1, 1.2], gap="medium")
        
        # ZONE 1: RACE CONFIGURATION
        with col_race:
            st.caption("🏁 RACE SETUP")
            sel_gp_mcda = st.selectbox("Circuit Selection", ["Monaco", "Spa", "Monza", "Silverstone", "Bahrain"], key="mcda_gp")
            
            # Physics Database
            track_db = {
                "Monaco": (78, 3.3, 74.0, 24.0, 3.5, 2.0),
                "Spa":    (44, 7.0, 106.0, 22.0, 0.8, 4.0),
                "Monza":  (53, 5.8, 81.0, 24.0, 1.2, 3.0),
                "Silverstone": (52, 5.9, 87.0, 20.0, 1.4, 4.0),
                "Bahrain": (57, 5.4, 92.0, 22.0, 1.5, 3.0)
            }
            c_laps, c_len, c_base, c_pit, c_delta, c_ab = track_db.get(sel_gp_mcda, (57, 5.4, 90.0, 22.0, 1.5, 3.0))
            
            # Compact Layout for Grid/Laps
            c1, c2 = st.columns(2)
            u_grid = c1.number_input("Grid Pos", 1, 20, 10, help="Starting Grid Position")
            u_laps = c2.number_input("Total Laps", 1, 100, c_laps)

        # ZONE 2: LIVE CONDITIONS
        with col_cond:
            st.caption("🌦️ LIVE PHYSICS")
            
            # Physics Row
            p1, p2 = st.columns(2)
            u_temp = p1.number_input("Track °C", 10, 60, 35)
            u_fuel = p2.number_input("Fuel Eff.", 0.01, 0.1, 0.035, format="%.3f", help="Seconds lost per kg of fuel")
            
            # Probabilities Row
            st.markdown("---") 
            u_rain = st.slider("🌧️ Rain Probability", 0.0, 1.0, 0.0)
            u_sc = st.slider("⚠️ Safety Car Risk", 0.0, 1.0, 0.2)

        # ZONE 3: AI BRAIN
        with col_ai:
            st.caption("🧠 AI STRATEGY BIAS")
            st.info("Tune the AI's decision-making logic below.")
            
            # K-Factors with Tooltips
            k_safe = st.slider("Safety Weight (Avoid DNF) = K2", 0.1, 5.0, 2.0, help="Higher = AI prioritizes avoiding tire failure over speed.")
            k_traf = st.slider("Traffic Weight (Clean Air) = K3", 0.1, 5.0, 3.0, help="Higher = AI aggressively avoids traffic.")
            k_rob  = st.slider("Flexibility Weight (Options) = K4", 0.1, 5.0, 2.5, help="Higher = AI prefers strategies that can react to Safety Cars.")

    # ------------------------------------------------------------------
    # 2. ENGINE LOGIC & CALCULATIONS
    # ------------------------------------------------------------------
    circuit_p = CircuitParams(sel_gp_mcda, u_laps, c_len, c_base, c_pit, c_delta, c_ab)
    car_p = CarParams(110.0, 1.8, u_fuel, 0.5)
    env_p = EnvParams(u_temp, u_rain, u_sc, 0.2, 20, u_grid)
    tires_p = {
        'SOFT': TireParams('SOFT', 0.0, 0.12, 20, 2.0),
        'MEDIUM': TireParams('MEDIUM', 0.5, 0.08, 30, 2.0),
        'HARD': TireParams('HARD', 1.1, 0.04, 45, 2.0)
    }
    k_factors = {'safety': k_safe, 'traffic': k_traf, 'robust': k_rob}
    
    engine = F1DecisionEngine(circuit_p, car_p, env_p, tires_p, k_factors)
    weights = engine.calculate_dynamic_weights()
    
    # CRITICAL: Always re-calculate utility on load to prevent KeyError
    if st.session_state.saved_strategies:
        st.session_state.saved_strategies = engine.calculate_utility(st.session_state.saved_strategies)

    # ------------------------------------------------------------------
    # 3. LIVE WEIGHTS DISPLAY
    # ------------------------------------------------------------------
    st.markdown("##### ⚖️ Current AI Priorities")
    w_cols = st.columns(4)
    w_labels = ["SPEED (α1)", "SAFETY (α2)", "TRAFFIC (α3)", "FLEX (α4)"]
    w_vals = [weights['alpha_1'], weights['alpha_2'], weights['alpha_3'], weights['alpha_4']]
    
    for i, col in enumerate(w_cols):
        val_pct = w_vals[i] * 100
        # Color coding for visual emphasis
        border_color = "#3b82f6" if i == 0 else "#64748b" 
        col.markdown(f"""
        <div style="text-align:center; padding:8px; background:#1e293b; border-radius:6px; border-left: 4px solid {border_color};">
            <div style="font-size:10px; color:#94a3b8; font-weight:bold;">{w_labels[i]}</div>
            <div style="font-size:20px; font-weight:800; color:#f8fafc;">{val_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4. WORKSPACE: BUILDER & RESULTS
    # ------------------------------------------------------------------
    col_build, col_res = st.columns([1, 1.5])
    
    # --- LEFT: STRATEGY BUILDER ---
    with col_build:
        st.markdown("##### 🛠️ Strategy Builder")
        with st.container(border=True):
            # Input Row
            bc1, bc2 = st.columns([1.5, 1])
            s_comp = bc1.selectbox("Compound", ["SOFT", "MEDIUM", "HARD"])
            s_laps = bc2.number_input("Laps", 1, u_laps, 20)
            
            if st.button("➕ Add Stint", use_container_width=True): 
                st.session_state.user_stints.append({'compound': s_comp, 'laps': s_laps})
            
            # Live Preview
            if st.session_state.user_stints:
                st.markdown("---")
                st.caption("Strategy Preview:")
                
                # Gantt Chart
                stint_df = pd.DataFrame(st.session_state.user_stints)
                stint_df['Stint'] = range(1, len(stint_df) + 1)
                stint_df['Color'] = stint_df['compound'].map({'SOFT':'#ef4444', 'MEDIUM':'#eab308', 'HARD':'#f8fafc'})
                
                fig_gantt = px.bar(stint_df, x="laps", y="Stint", orientation='h', 
                                  color="Color", color_discrete_map="identity", text="compound")
                fig_gantt.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                fig_gantt.update_traces(textfont_color='black')
                st.plotly_chart(fig_gantt, use_container_width=True)
                
                # Logic
                tot = sum(s['laps'] for s in st.session_state.user_stints)
                if tot == u_laps:
                    if st.button("💾 SAVE PLAN", type="primary", use_container_width=True):
                        strat = engine.evaluate_strategy(st.session_state.user_stints)
                        st.session_state.saved_strategies.append(strat)
                        # Normalize immediately
                        st.session_state.saved_strategies = engine.calculate_utility(st.session_state.saved_strategies)
                        st.session_state.user_stints = []
                        st.rerun()
                elif tot > u_laps:
                    st.error(f"⚠️ {tot}/{u_laps} Laps (Too Long)")
                else:
                    st.info(f"Drafting: {tot}/{u_laps} Laps")
                
                if st.button("Clear Draft"): st.session_state.user_stints = []

        st.markdown("##### 🤖 AI Optimizer")
        with st.container(border=True):
            st.info("The AI will simulate 100+ valid strategies based on your K-Factors above.")
            if st.button("✨ GENERATE OPTIMAL", type="primary", use_container_width=True):
                with st.spinner("Running Monte Carlo Simulation..."):
                    best = engine.generate_optimal_strategies(n_gen=100)
                    st.session_state.saved_strategies.extend(best)
                    st.session_state.saved_strategies = engine.calculate_utility(st.session_state.saved_strategies)
                    st.rerun()

    # --- RIGHT: DECISION MATRIX ---
    with col_res:
        st.markdown("##### 📊 Strategy Matrix")
        
        if st.session_state.saved_strategies:
            sorted_strats = sorted(st.session_state.saved_strategies, key=lambda x: x['Utility_Score'])
            
            # Table View
            table_data = []
            for s in sorted_strats:
                table_data.append({
                    "Strategy": s['Name'],
                    "Time": f"{int(s['C1_Time'])}s",
                    "Risk": f"{s['C2_Risk']:.0f}%",
                    "Traffic": f"{s['C3_Traffic']:.1f}",
                    "Score": f"{s['Utility_Score']:.2f}"
                })
            
            st.dataframe(
                pd.DataFrame(table_data), 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Utility Score (Lower=Better)", format="%.2f", min_value=0, max_value=10)
                }
            )
            
            # Analysis Chart
            st.markdown("##### 🔍 Performance Profile")
            best = sorted_strats[0]
            
            # Robust check for normalized scores
            if 'Normalized_Scores' in best:
                scores = best['Normalized_Scores']
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[scores['C1'], scores['C2'], scores['C3'], scores['C4']],
                    theta=['Speed Cost', 'Tire Risk', 'Traffic', 'Inflexibility'],
                    fill='toself',
                    name=best['Name'],
                    line_color='#10b981'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10], showticklabels=False)),
                    height=300,
                    margin=dict(l=40, r=40, t=20, b=20),
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            if st.button("🗑️ Clear Matrix"): 
                st.session_state.saved_strategies = []
                st.rerun()
        else:
            st.info("👈 Use the tools on the left to populate the decision matrix.")


# ==============================================================================
# TAB 2: LIVE TRACK MAP (GHOST CAR FIX)
# ==============================================================================
with tabs[1]:
    if st.session_state.dm.session:
        # --- 1. ROBUST DATA LOADING ---
        res_df = st.session_state.res.get_results(st.session_state.dm.session)
        fast_lap = st.session_state.res.get_fastest_lap_comparison(st.session_state.dm.session)
        
        try: weather_temp = st.session_state.dm.session.weather_data.iloc[-1]['TrackTemp']
        except: weather_temp = 0.0
        
        # --- 2. HERO METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        
        leader_txt = res_df.iloc[0]['Abbreviation'] if res_df is not None and not res_df.empty else "---"
        m1.markdown(f"<div class='metric-container'><div class='metric-label'>RACE LEADER</div><div class='metric-value' style='color:#FACC15'>🥇 {leader_txt}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-container'><div class='metric-label'>FASTEST</div><div class='metric-value' style='color:#D946EF'>LAP 42</div></div>", unsafe_allow_html=True) 
        m3.markdown(f"<div class='metric-container'><div class='metric-label'>TRACK</div><div class='metric-value' style='color:#EF4444'>{weather_temp:.1f}°C</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-container'><div class='metric-label'>STATUS</div><div class='metric-value' style='color:#22C55E'>LIVE</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- 3. MAIN INTERFACE ---
        col_map, col_standings = st.columns([3, 1])
        
        with col_map:
            st.markdown("##### 🛰️ GPS Tracking")
            
            # A. STATIC TRACK TRACE
            track_trace = None
            try:
                # Use leader's best lap for the grey road line
                ref_lap = st.session_state.dm.session.laps.pick_fastest()
                if ref_lap is None: ref_lap = st.session_state.dm.session.laps.pick_wo_box().iloc[0]
                
                pos_data = ref_lap.get_telemetry()
                track_trace = go.Scatter(
                    x=pos_data['X'], y=pos_data['Y'],
                    mode='lines',
                    line=dict(color='#334155', width=12), # The Tarmac
                    name='Circuit', hoverinfo='skip'
                )
            except: pass

            # B. DYNAMIC POSITIONS
            replay_df = st.session_state.res.generate_replay_frame(st.session_state.dm.session)
            
            if replay_df is not None and not replay_df.empty:
                # 0-100% Slider
                progress = st.slider("⏱️ Replay Scrubber", 0, 100, 0, label_visibility="collapsed")
                
                # Filter current step
                frame = replay_df[replay_df['Step'] == progress]
                
                # --- PLOT ---
                fig_map = go.Figure()
                
                # Layer 1: Road
                if track_trace: fig_map.add_trace(track_trace)
                
                # Layer 2: Drivers (Now guarantees all drivers present)
                team_colors = {
                    'Red Bull Racing': '#3671C6', 'Mercedes': '#27F4D2', 'Ferrari': '#E8002D',
                    'McLaren': '#FF8000', 'Aston Martin': '#229971', 'Alpine': '#0093CC',
                    'Williams': '#64C4FF', 'Haas F1 Team': '#B6BABD', 'Kick Sauber': '#52E252', 'RB': '#6692FF'
                }
                
                for i, row in frame.iterrows():
                    t_col = team_colors.get(row['Team'], '#FFFFFF')
                    fig_map.add_trace(go.Scatter(
                        x=[row['X']], y=[row['Y']],
                        mode='markers+text',
                        marker=dict(size=18, color=t_col, line=dict(width=1, color='white')),
                        text=row['Driver'], textposition='middle center',
                        textfont=dict(size=9, color='black', family='Arial Black'),
                        name=row['Driver'], hoverinfo='name'
                    ))
                
                fig_map.update_layout(
                    height=600, template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                    yaxis=dict(visible=False),
                    showlegend=False, margin=dict(l=0,r=0,t=0,b=0)
                )
                st.plotly_chart(fig_map, use_container_width=True)
                
            else:
                st.info("Loading Simulation Data... (Physics Engine Warmup)")

        # --- 4. STANDINGS TOWER ---
        with col_standings:
            st.markdown("##### ⏱️ Standings")
            if res_df is not None:
                st.dataframe(
                    res_df,
                    hide_index=True,
                    height=700,
                    use_container_width=True,
                    column_config={
                        "Position": st.column_config.TextColumn("#", width="small"),
                        "Abbreviation": st.column_config.TextColumn("Driver", width="small"),
                        "GapToLeader": st.column_config.TextColumn("Gap", width="medium"),
                        "TeamName": st.column_config.TextColumn("Team", width="medium"),
                    }
                )
    else:
        st.info("⚠️ Connect to a session in the Sidebar.")
# ==============================================================================
# TAB 3: BATTLE (HEAD-TO-HEAD)
# ==============================================================================
with tabs[2]:
    if st.session_state.dm.session:
        drivers = st.session_state.dm.get_driver_list()
        d_list = [d[0] for d in drivers]
        c1, c2, c3 = st.columns([1, 0.2, 1])
        with c1: d1 = st.selectbox("DRIVER A", d_list, index=0, key="bat_d1")
        with c2: st.markdown("<h2 style='text-align: center; color: #94a3b8;'>VS</h2>", unsafe_allow_html=True)
        with c3: d2 = st.selectbox("DRIVER B", d_list, index=1, key="bat_d2")
        
        if d1 and d2:
            data1 = st.session_state.dm.get_clean_telemetry(d1)
            data2 = st.session_state.dm.get_clean_telemetry(d2)
            if data1 and data2:
                t1, t2 = data1['telemetry'], data2['telemetry']
                st.markdown("##### 🗺️ Track Dominance")
                sectors = st.session_state.analyst.calculate_mini_sectors(t1, t2, n_sectors=50)
                if not sectors.empty:
                    fig_dom = go.Figure()
                    fig_dom.add_trace(go.Scatter(x=t1['X'], y=t1['Y'], mode='lines', line=dict(color='#334155', width=10), hoverinfo='skip'))
                    colors = {1: '#3b82f6', 2: '#ef4444'}
                    fig_dom.add_trace(go.Scatter(x=sectors['X'], y=sectors['Y'], mode='markers', marker=dict(color=sectors['Winner'].map(colors), size=8), hoverinfo='skip'))
                    fig_dom.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", xaxis=dict(visible=False, scaleanchor="y", scaleratio=1), yaxis=dict(visible=False), showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig_dom, use_container_width=True)
                
                st.markdown("##### 📉 Telemetry Analysis")
                fig_spd = go.Figure()
                fig_spd.add_trace(go.Scatter(x=t1['Distance'], y=t1['Speed'], name=d1, line=dict(color='#3b82f6')))
                fig_spd.add_trace(go.Scatter(x=t2['Distance'], y=t2['Speed'], name=d2, line=dict(color='#ef4444')))
                fig_spd.update_layout(height=250, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1e293b", margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(showticklabels=False), yaxis=dict(title="Speed", gridcolor='#334155'))
                st.plotly_chart(fig_spd, use_container_width=True)
    else: st.info("Connect to session")

# ==============================================================================
# REMAINING TABS (4-9)
# ==============================================================================
if st.session_state.dm.session:
    drivers = st.session_state.dm.get_driver_list()
    d_opts = [d[0] for d in drivers]

    with tabs[3]: # Stints
        d_stint = st.selectbox("Driver", d_opts, key="st_d")
        if st.button("Analyze Stints", type="primary"):
            stints = st.session_state.analyst.analyze_stint(st.session_state.dm.session, d_stint)
            if stints:
                fig_s = go.Figure()
                for s in stints:
                    df = s['Lap_Data']
                    color = "#ef4444" if s['Compound']=='SOFT' else "#eab308" if s['Compound']=='MEDIUM' else "#f8fafc"
                    fig_s.add_trace(go.Scatter(x=df['LapNumber'], y=df['LapTime'].dt.total_seconds(), mode='markers+lines', name=f"{s['Compound']}", line=dict(color=color)))
                st.plotly_chart(fig_s, use_container_width=True)

    with tabs[4]: # Concept
        if st.button("Scan Grid"):
            traits = st.session_state.analyst.calculate_setup_traits(st.session_state.dm.session)
            st.session_state['traits'] = traits
        if 'traits' in st.session_state:
            fig_c = px.scatter(st.session_state['traits'], x="Top_Speed", y="Cornering_Speed", text="Driver", color="Team")
            st.plotly_chart(fig_c, use_container_width=True)

    with tabs[5]: # Race Strat
        from core.strategy_engine import TrafficOracle
        oracle = TrafficOracle()
        trace = oracle.get_race_trace(st.session_state.dm.session)
        if trace is not None:
            st.plotly_chart(px.line(trace, x="LapNumber", y="GapToLeader", color="Driver"), use_container_width=True)

    with tabs[6]: # Telemetry
        c1, c2 = st.columns(2)
        with c1: td1 = st.selectbox("Focus", d_opts, key="t1")
        with c2: td2 = st.selectbox("Ref", d_opts, key="t2")
        if td1 and td2:
            data1 = st.session_state.dm.get_clean_telemetry(td1)
            data2 = st.session_state.dm.get_clean_telemetry(td2)
            if data1 and data2:
                t1, t2 = data1['telemetry'], data2['telemetry']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t1['Distance'], y=t1['Speed'], name=td1))
                fig.add_trace(go.Scatter(x=t2['Distance'], y=t2['Speed'], name=td2, line=dict(dash='dot')))
                st.plotly_chart(fig)

    with tabs[7]: # ML
        if st.button("Run Clusters"):
            laps = st.session_state.ml.cluster_laps(st.session_state.dm.laps)
            st.plotly_chart(px.scatter(laps, x="LapNumber", y="LapTimeSec", color="Lap_Type"))

    with tabs[8]: # Matrix
        c1, c2 = st.columns(2)
        with c1: p1 = st.selectbox("Driver", d_opts, key="p1")
        with c2: p2 = st.selectbox("Ref", d_opts, key="p2")
        if p1 and p2:
            il1 = st.session_state.analyst.calculate_ideal_lap(st.session_state.dm.session, p1)
            if il1: st.metric("Theoretical Best", f"{il1['Theoretical_Lap']:.3f}s")
else:
    for i in range(1, 9):
        with tabs[i]: st.info("Connect to session")