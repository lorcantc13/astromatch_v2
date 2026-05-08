from math import erf, sqrt, pi

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. APP CONFIG ---
st.set_page_config(page_title="AstroMatch V2", layout="wide")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        analogues = pd.read_csv('analogues_v2.csv')
        targets = pd.read_csv('targets_v2.csv')
        return analogues, targets
    except FileNotFoundError:
        st.error("Data files not found. Ensure 'analogues_v2.csv' and 'targets_v2.csv' are in the directory.")
        return pd.DataFrame(), pd.DataFrame()

analogues_df, targets_df = load_data()

# --- 3. SCORING ENGINE (ASYMMETRIC TOLERANCE ENVELOPE + LETHAL CLIFF) ---
#
# Lethal-limit envelopes for known terrestrial life. Used as hard cliffs:
# any portion of the target range outside these bounds contributes zero
# regardless of how close the analogue site comes. Sources are summarised
# in documentation/scoring_theory.md. None means "no universal limit".
#
# Values were sanity-checked against the cited literature (May 2026).
# They are deliberately conservative and intended to be reviewed by a
# domain expert before any scientific publication.
LETHAL_LIMITS = {
    "T":     (-25.0, 122.0),   # Cold: cryobrine/perchlorate literature (Toner et al. 2014).
                               # Hot: Methanopyrus kandleri strain 116 at 122 C / 20 MPa
                               #      (Takai et al. 2008, PNAS 105:10949).
    "Sal":   (None, None),     # No universal upper salinity cap; halophiles grow to NaCl saturation.
    "pH":    (0.0, 12.5),      # Acid: Picrophilus torridus growth around pH 0 (Schleper et al. 1995).
                               # Alkali: Serpentinimonas maccroryi B1 at pH 12.5 (Suzuki et al. 2021,
                               #         IJSEM; original genus described in Suzuki et al. 2014, Nat Commun).
    "Pres":  (0.0, 1100.0),    # Active metabolism observed up to ~1060 MPa for E. coli / Shewanella
                               # (Sharma et al. 2002, Science 295:1514). Cell viability extends to
                               # ~1600 MPa in fluid inclusions; 1100 MPa is the conservative
                               # active-growth cap.
    "Iso":   (None, None),     # Ordinal score; no biological universal.
    "Redox": (None, None),
}

def calculate_suitability(site_min, site_max, target_min, target_max,
                          lethal_min=None, lethal_max=None):
    """Score how well an analogue site's range *contains* a target range.

    Models the observed analogue range [site_min, site_max] as the inferred
    tolerance envelope of life that survives there: full credit for target
    conditions inside the envelope, Gaussian-decaying credit just outside,
    and a hard zero beyond the universal-life lethal limits.

    The score is asymmetric on purpose. The biological question is "could
    the analogue's organism survive these target conditions?", not "are
    these two ranges similar?" — so a wide-tolerance analogue covering a
    narrow target should score 1.0, while the reverse should not.
    """
    if any(pd.isna(v) for v in (site_min, site_max, target_min, target_max)):
        return None

    # Hard lethal short-circuit: target entirely outside life's envelope.
    if lethal_min is not None and target_max < lethal_min:
        return 0.0
    if lethal_max is not None and target_min > lethal_max:
        return 0.0

    # Clip target to the viable envelope; mass outside contributes 0 to the score.
    eff_t_min = target_min if lethal_min is None else max(target_min, lethal_min)
    eff_t_max = target_max if lethal_max is None else min(target_max, lethal_max)
    if eff_t_max <= eff_t_min:
        return 0.0
    viable_fraction = (eff_t_max - eff_t_min) / max(target_max - target_min, 1e-9)

    # Soft-edge bandwidth: 25% of analogue range, floored to avoid degenerate
    # zero widths from single-point site measurements.
    delta = 0.25 * max(site_max - site_min, 1.0)

    # μ(x) is 1 on [site_min, site_max] and decays as exp(-((dist)/delta)^2)
    # outside it. Compute the mean of μ over [eff_t_min, eff_t_max] in closed
    # form: inside contributes its overlap length; the two outside tails are
    # erf differences of the Gaussian decay.
    inside_lo = max(eff_t_min, site_min)
    inside_hi = min(eff_t_max, site_max)
    inside_area = max(inside_hi - inside_lo, 0.0)

    half_sqrt_pi_delta = (sqrt(pi) / 2.0) * delta
    def _gauss_tail_integral(a, b, centre):
        if b <= a:
            return 0.0
        return half_sqrt_pi_delta * (erf((b - centre) / delta) - erf((a - centre) / delta))

    below_area = _gauss_tail_integral(eff_t_min, min(eff_t_max, site_min), site_min)
    above_area = _gauss_tail_integral(max(eff_t_min, site_max), eff_t_max, site_max)

    mean_membership = (inside_area + below_area + above_area) / (eff_t_max - eff_t_min)
    return float(np.clip(mean_membership * viable_fraction, 0.0, 1.0))

# --- 4. SIDEBAR: WEIGHTING & VISUALS ---
st.sidebar.header("🎯 Importance Weights")
st.sidebar.info("Toggle parameters and adjust influence (1-10)")
st.sidebar.markdown("📚 **[Read the AstroMatch Documentation](https://github.com/lorcantc13/astromatch_v2/tree/main/documentation)**")
st.sidebar.write("") # Adds a little space before the toggles start

params_config = {
    "Temperature": {"color": "#EF553B", "default": 5, "col_prefix": "T"},
    "Salinity": {"color": "#F5F5F5", "default": 5, "col_prefix": "Sal"},
    "pH": {"color": "#00CC96", "default": 5, "col_prefix": "pH"},
    "Pressure": {"color": "#AB63FA", "default": 5, "col_prefix": "Pres"},
    "Isolation": {"color": "#6ab7f1", "default": 5, "col_prefix": "Iso"},
    "Redox Potential": {"color": "#FF8C00", "default": 5, "col_prefix": "Redox"}
}

user_weights = {}
active_params = []

for name, info in params_config.items():
    # Create two columns: one for the title (wider), one for the toggle (narrower)
    col_title, col_toggle = st.sidebar.columns([4, 1])
    
    with col_title:
        st.markdown(f"**<span style='color:{info['color']}'>{name}</span>**", unsafe_allow_html=True)
        if "help" in info:
            st.caption(f"ℹ️ {info['help']}")
            
    with col_toggle:
        # Use an empty string for the label and collapse visibility to remove the text entirely
        is_on = st.toggle(" ", value=True, key=f"tog_{name}", label_visibility="collapsed")
    
    # Put the slider directly underneath
    val = st.sidebar.slider(f"{name} Weight", 1, 10, info['default'], label_visibility="collapsed", key=f"sld_{name}", disabled=not is_on)
    
    st.sidebar.write("") # Adds a tiny bit of breathing room between parameters
    
    if is_on:
        user_weights[name] = val
        active_params.append(name)

# Dynamic Donut Chart
if user_weights:
    weights_df = pd.DataFrame({
        "Parameter": list(user_weights.keys()),
        "Weight": list(user_weights.values())
    })
    fig_donut = px.pie(
        weights_df, values='Weight', names='Parameter', hole=0.5, color='Parameter',
        color_discrete_map={k: v['color'] for k, v in params_config.items()}
    )
    fig_donut.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.sidebar.plotly_chart(fig_donut, use_container_width=True)
else:
    st.sidebar.warning("Please enable at least one parameter.")

# --- 5. MAIN INTERFACE ---
st.title("🪐 AstroMatch MCDA Tool")

col_a, col_b = st.columns(2)
with col_a:
    body_choice = st.selectbox("1. Select Target Body", ["Enceladus", "Europa", "Mars"])

with col_b:
    if body_choice == "Enceladus" and not targets_df.empty:
        env_list = targets_df['Target_env'].unique().tolist()
        target_env = st.selectbox("2. Select Environment", env_list)
    else:
        st.selectbox("2. Select Environment", ["🚧 Coming Soon..."], disabled=True)
        target_env = None

with st.expander("🛠 Advanced Options"):
    st.selectbox("Select Organism (Preset Weights)", ["🚧 Coming Soon..."], disabled=True)
    st.file_uploader("Import Custom Analogue Site", disabled=True)

st.divider()

# --- 6. EXECUTION & OUTPUT ---
if st.button("🚀 Run Analysis") and target_env and user_weights:
    target_data = targets_df[targets_df['Target_env'] == target_env].iloc[0]
    results = []
    
    w_sum_total = sum(user_weights.values())

    for _, site in analogues_df.iterrows():
        site_fits = {}
        site_rels = {}
        active_site_weights = {}
        flags = []
        
        # Calculate Fits and Handle Missing Data
        for param in active_params:
            prefix = params_config[param]['col_prefix']
            
            # Identify columns (handling scalar rubrics vs range data)
            min_col = f"{prefix}_min" if f"{prefix}_min" in site else f"{prefix}_score"
            max_col = f"{prefix}_max" if f"{prefix}_max" in site else f"{prefix}_score"
            rel_col = f"{prefix}_rel"
            
            t_min_col = f"{prefix}_min" if f"{prefix}_min" in target_data else f"{prefix}_score"
            t_max_col = f"{prefix}_max" if f"{prefix}_max" in target_data else f"{prefix}_score"

            lethal_lo, lethal_hi = LETHAL_LIMITS.get(prefix, (None, None))
            fit = calculate_suitability(
                site[min_col], site[max_col],
                target_data[t_min_col], target_data[t_max_col],
                lethal_min=lethal_lo, lethal_max=lethal_hi,
            )
            
            if fit is not None:
                site_fits[param] = fit
                site_rels[param] = site.get(rel_col, 1) # Default to 1 if rel missing
                active_site_weights[param] = user_weights[param]
            else:
                # Missing Data Logic
                weight_pct = user_weights[param] / w_sum_total
                if weight_pct > 0.05:
                    flags.append(f"⚠️ Missing {param} data (>5% weight)")
                site_fits[param] = "N/A"
                site_rels[param] = "N/A"

        # Liebig aggregation: weighted *geometric* mean of per-parameter fits.
        # A weighted arithmetic mean lets a perfect score on five parameters
        # mask a lethal sixth, which contradicts ecology's Law of the Minimum.
        # The geometric mean is the joint survival probability under
        # independence and collapses toward zero whenever any single fit does.
        actual_w_sum = sum(active_site_weights.values())
        if actual_w_sum > 0:
            log_score = 0.0
            for p, w in active_site_weights.items():
                # Floor at epsilon so log() is defined on hard-zero fits while
                # still propagating their dominance through the product.
                fit = max(float(site_fits[p]), 1e-3)
                log_score += (w / actual_w_sum) * np.log(fit)
            final_score = float(np.exp(log_score))
        else:
            final_score = 0.0
            
        # Calculate Confidence Score & Quality Flags
        if active_site_weights:
            fits_for_conf = [site_fits[p] for p in active_site_weights]
            rels_for_conf = [site_rels[p] for p in active_site_weights]
            
            if sum(fits_for_conf) > 0:
                conf_score = sum(f * r for f, r in zip(fits_for_conf, rels_for_conf)) / sum(fits_for_conf)
            else:
                conf_score = np.mean(rels_for_conf)
                
            # Check for high fit, low rel, high weight
            for p in active_site_weights:
                if site_fits[p] > 0.7 and site_rels[p] == 1 and (active_site_weights[p]/actual_w_sum) > 0.2:
                    flags.append(f"⚠️ Low Reliability on heavy driver: {p}")
        else:
            conf_score = 0.0
            
        alert_str = " | ".join(flags) if flags else "✅ Reliable"

        # Store result
        res_dict = {
            "Site": site['Site'],
            "Suitability": round(final_score, 4),
            "Confidence": round(conf_score, 2),
            "Alerts": alert_str,
            "lat": site.get('lat', None),
            "lon": site.get('lon', None)
        }
        
        # Store individual fits/rels for detailed view
        for p in params_config.keys():
            res_dict[f"{p} Fit"] = site_fits.get(p, "Off/NA")
            res_dict[f"{p} Rel"] = site_rels.get(p, "Off/NA")
            
        results.append(res_dict)

    res_df = pd.DataFrame(results).sort_values("Suitability", ascending=False).reset_index(drop=True)
    res_df.index += 1 # 1-based ranking

    st.session_state['res_df'] = res_df
    st.session_state['target_env'] = target_env

# --- 7. RESULTS DASHBOARD ---
if 'res_df' in st.session_state:
    res_df = st.session_state['res_df']
    
    st.subheader("🏆 Ranked Shortlist")
    
    # Display Top 5
    display_cols = ['Site', 'Suitability', 'Confidence', 'Alerts']
    st.dataframe(
        res_df.head(5)[display_cols].style.background_gradient(subset=['Suitability'], cmap="Blues"), 
        use_container_width=True
    )
    
    with st.expander("View all sites"):
        st.dataframe(res_df[display_cols], use_container_width=True)
        
    st.divider()
    
    # --- SITE PROFILE ---
    st.subheader("🔍 Detailed Site Profile")
    selected_site = st.selectbox("Select a site to inspect:", res_df['Site'].tolist())
    
    site_data = res_df[res_df['Site'] == selected_site].iloc[0]
    
    # 1. Dynamic Verdict
    strong, mod, weak = [], [], []
    for p in active_params:
        val = site_data[f"{p} Fit"]
        if val != "N/A" and val != "Off/NA":
            if val >= 0.7: strong.append(p)
            elif val >= 0.4: mod.append(p)
            else: weak.append(p)
    
    verdict = f"**{selected_site}** is an analogue match of **{site_data['Suitability']*100:.1f}%**. "
    if strong: verdict += f"It scores strongly on {', '.join(strong)}. "
    if mod: verdict += f"It scores moderately on {', '.join(mod)}. "
    if weak: verdict += f"It has weaker fidelity regarding {', '.join(weak)}."
    
    st.info(verdict)
    
    # --- Profile Layout ---
    st.write("### Analogue Footprint vs Target")
    
    # 1. RADAR CHART (Full Width / Prominent)
    categories = active_params
    r_vals = [site_data[f"{p} Fit"] if isinstance(site_data[f"{p} Fit"], float) else 0 for p in categories]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=[1]*len(categories), theta=categories, fill='toself', name='Target', line_color='gold'))
    fig_radar.add_trace(go.Scatterpolar(r=r_vals, theta=categories, fill='toself', name=selected_site, line_color='cyan'))
    
    # Make it taller and move the legend to the bottom so it doesn't squash the chart
    fig_radar.update_layout(
        height=500, 
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="radar")

    st.divider()

    # 2. TABLE AND MAP (Side-by-Side underneath)
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("### Parameter Breakdown")
        breakdown_data = []
        for p in active_params:
            breakdown_data.append({
                "Parameter": p,
                "Fidelity": site_data[f"{p} Fit"],
                "Data Quality (Rel)": site_data[f"{p} Rel"]
            })
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

    with c2:
        st.write("### Global Location")
        if pd.notna(site_data['lat']) and pd.notna(site_data['lon']):
            map_df = pd.DataFrame({"lat": [site_data['lat']], "lon": [site_data['lon']], "Site": [selected_site]})
            fig_map = px.scatter_geo(map_df, lat="lat", lon="lon", hover_name="Site", projection="natural earth")
            fig_map.update_traces(marker=dict(size=12, color="red"))
            fig_map.update_geos(showcountries=True, countrycolor="RebeccaPurple")
            fig_map.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_map, use_container_width=True, key="map")
        else:
            st.warning("No coordinate data (lat/lon) available for this site.")
