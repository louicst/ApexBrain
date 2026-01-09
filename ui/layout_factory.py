import streamlit as st

def card_header(title, context_id):
    """
    Renders the Header of a Bento Tile with the 'Ask ML' button.
    Args:
        title: The title of the card (e.g. "TYRE DEGRADATION")
        context_id: Unique ID to trigger specific ML logic for this card.
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{title}**")
    with col2:
        # The 'Ask ML' Button logic
        # We use a distinct key for every button so Streamlit tracks them separately
        if st.button("🧠 Ask ML", key=f"btn_ml_{context_id}", help="Run Contextual AI Analysis"):
            st.session_state['ml_trigger'] = context_id
            st.session_state['show_ml_sidebar'] = True

def render_ml_sidebar(ml_engine, data_manager):
    """
    The Contextual AI Sidebar that slides out when 'Ask ML' is clicked.
    """
    if st.session_state.get('show_ml_sidebar', False):
        with st.sidebar:
            st.markdown("---")
            st.subheader("🧠 Cortex AI Insight")
            
            context = st.session_state.get('ml_trigger', 'generic')
            
            with st.spinner(f"Analyzing {context} data..."):
                # MOCK RESPONSE - In production, this calls ml_engine.analyze(context)
                if context == "tyre_deg":
                    st.success("Insight Generated")
                    st.info("**Observation:** Hard compound degradation is 15% lower than expected.")
                    st.warning("**Prediction:** Crossover point to Softs is Lap 42.")
                elif context == "telemetry":
                    st.success("Insight Generated")
                    st.info("**Technique:** Driver A is braking 5m later in Turn 4 but compromising exit speed by 3 km/h.")
                else:
                    st.info("Analysis complete. No anomalies detected.")
            
            if st.button("Close Insight"):
                st.session_state['show_ml_sidebar'] = False
                st.rerun()