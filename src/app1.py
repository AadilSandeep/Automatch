import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommender import VehicleRecommender

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoMatch – Vehicle Recommender",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL STYLES ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base font */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit chrome completely */
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Kill default top padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar (dark navy) ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0c1628 0%, #0f1f3d 55%, #0d2847 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 290px !important;
    max-width: 320px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem 2rem !important;
}

/* Sidebar text, labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* Sidebar widget bg */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Sidebar radio */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important;
}

/* Sidebar select_slider */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding: 0 !important;
}

/* ── Main CTA button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    height: 46px !important;
    width: 100% !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.38) !important;
    transition: all 0.2s ease !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79,70,229,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Main area background ── */
.stApp { background: #f1f5f9 !important; }

/* ── Warning / info overrides ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.84rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD RECOMMENDER ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_recommender():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "cars.csv")
        return VehicleRecommender(data_path)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None

recommender = get_recommender()


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Brand + Preferences Form
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand header
    st.markdown("""
    <div style="margin-bottom:28px;">
      <div style="display:flex;align-items:center;gap:11px;margin-bottom:6px;">
        <div style="width:40px;height:40px;border-radius:10px;flex-shrink:0;
                    background:linear-gradient(135deg,#2563eb,#4f46e5);
                    display:flex;align-items:center;justify-content:center;
                    font-size:19px;box-shadow:0 3px 10px rgba(79,70,229,0.4);"></div>
        <div>
          <div style="font-size:19px;font-weight:800;color:#f1f5f9;
                      letter-spacing:-0.5px;line-height:1.1;">AutoMatch</div>
          <div style="font-size:11px;color:#475569;font-weight:400;margin-top:1px;">
              AI Vehicle Recommender</div>
        </div>
      </div>
      <p style="font-size:12px;color:#334155;line-height:1.6;margin:14px 0 0;">
          Tell us what you need. We'll find the perfect car from 500+ models.
      </p>
    </div>

    <div style="border-top:1px solid rgba(255,255,255,0.07);margin-bottom:22px;"></div>

    <div style="font-size:10px;font-weight:700;letter-spacing:1.6px;color:#334155;
                text-transform:uppercase;margin-bottom:14px;">Vehicle Preferences</div>
    """, unsafe_allow_html=True)

    budget = st.selectbox("Budget Range",  ["Low", "Mid", "High"],
                           help="Low ≈ under ₹8L · Mid ≈ ₹8–18L · High ≈ above ₹18L")
    fuel   = st.selectbox("Fuel Type",    ["Petrol", "Diesel", "CNG"])
    body   = st.selectbox("Body Type",    ["SUV", "Sedan", "Hatchback"])

    st.markdown("""
    <div style="border-top:1px solid rgba(255,255,255,0.07);margin:20px 0 18px;"></div>
    <div style="font-size:10px;font-weight:700;letter-spacing:1.6px;color:#334155;
                text-transform:uppercase;margin-bottom:14px;">Lifestyle & Usage</div>
    """, unsafe_allow_html=True)

    usage = st.selectbox(
        "Primary Use",
        ["City Commute", "Highway / Long Distance",
         "Family Trips", "Off-road / Rough Terrain", "Mixed"],
    )
    passengers = st.select_slider("Passengers Typically", options=["1–2", "3–4", "5+"])
    experience = st.radio(
        "Driving Experience",
        ["First-time Buyer", "Experienced"],
        horizontal=True,
    )
    priority = st.selectbox(
        "Top Priority",
        ["Balanced", "Fuel Efficiency", "Performance", "Safety"],
    )

    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    find_btn = st.button("Find My Car →")


# ── LOGIC ────────────────────────────────────────────────────────────────────────
if find_btn and recommender:
    lifestyle = {
        "usage":      usage,
        "passengers": passengers,
        "experience": experience,
        "priority":   priority,
    }
    specs = recommender.map_lifestyle_to_specs(lifestyle)
    prefs = {
        "budget":  budget,
        "fuel":    fuel,
        "body":    body,
        "mileage": specs.get("mileage", 15.0),
        "seats":   specs.get("seats",   5),
        "power":   specs.get("power",   100.0),
    }
    with st.spinner("Analysing your preferences…"):
        recs = recommender.recommend(prefs)
        st.session_state["recs"]   = recs
        st.session_state["prefs"]  = prefs


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Results
# ════════════════════════════════════════════════════════════════════════════════

# ── Page header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:28px;">
  <h1 style="font-size:27px;font-weight:800;color:#0f172a;
             letter-spacing:-0.6px;margin:0 0 4px;">
      Find Your Perfect Car</h1>
  <p style="font-size:14px;color:#64748b;margin:0;font-weight:400;">
      Recommendations powered by machine learning, personalised to your lifestyle.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ────────────────────────────────────────────────────────────────────
# Removed at user request to simplify UI


# ── Empty state ───────────────────────────────────────────────────────────────────
if "recs" not in st.session_state:
    st.markdown("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:18px;
                padding:72px 40px;text-align:center;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);">
      <div style="font-size:54px;margin-bottom:18px;">🚗</div>
      <div style="font-size:20px;font-weight:700;color:#1e293b;margin-bottom:8px;">
          No recommendations yet</div>
      <div style="font-size:14px;color:#94a3b8;max-width:360px;margin:0 auto;
                  line-height:1.7;">
          Use the panel on the left to set your budget, fuel type, and lifestyle
          preferences, then hit <strong style="color:#2563eb;">Find My Car</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    recs  = st.session_state["recs"]
    prefs = st.session_state["prefs"]

    if recs.empty:
        st.warning("No cars found matching your criteria. Try relaxing one or more filters.")

    else:
        n = len(recs)
        st.markdown(f"""
        <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:18px;">
            Results
            <span style="background:#eff6ff;color:#2563eb;font-size:12px;
                         font-weight:600;padding:3px 10px;border-radius:6px;
                         margin-left:8px;">{n} match{'es' if n != 1 else ''}</span>
        </div>
        """, unsafe_allow_html=True)

        for rank, (_, row) in enumerate(recs.iterrows(), start=1):
            score   = row["Similarity_Score"]
            price   = row["Ex-Showroom_Price"]
            mileage = row["Mileage"]
            power   = row["Power"]
            fuel_t  = row["Fuel_Type"]
            make    = row["Make"]
            model   = row["Model"]
            variant = row.get("Variant", "")

            # explanation
            if hasattr(recommender, "get_explanation"):
                why = recommender.get_explanation(row, prefs)
            else:
                why = (f"This {fuel_t} {body} fits your {budget.lower()} budget "
                       f"and {usage.lower()} usage pattern well.")

            # badge colour based on match score
            if score >= 0.85:
                b_bg, b_bd, b_tx = "#ecfdf5", "#86efac", "#15803d"
            elif score >= 0.65:
                b_bg, b_bd, b_tx = "#fefce8", "#fde047", "#92400e"
            else:
                b_bg, b_bd, b_tx = "#f8fafc", "#cbd5e1", "#475569"

            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;
                        padding:26px 28px;margin-bottom:16px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.05);
                        position:relative;overflow:hidden;">

              <!-- accent top bar -->
              <div style="position:absolute;inset:0 0 auto 0;height:3px;
                          background:linear-gradient(90deg,#2563eb,#4f46e5);
                          border-radius:16px 16px 0 0;"></div>

              <!-- rank pill -->
              <div style="position:absolute;top:16px;left:0;
                          background:linear-gradient(135deg,#2563eb,#4f46e5);
                          color:#fff;font-size:11px;font-weight:700;
                          padding:4px 13px 4px 11px;border-radius:0 5px 5px 0;">
                  #{rank}</div>

              <!-- header -->
              <div style="display:flex;justify-content:space-between;
                          align-items:flex-start;margin:18px 0 18px;">
                <div>
                  <div style="font-size:11px;font-weight:700;letter-spacing:1.2px;
                              text-transform:uppercase;color:#2563eb;margin-bottom:2px;">
                      {make}</div>
                  <div style="font-size:22px;font-weight:800;color:#0f172a;
                              letter-spacing:-0.4px;line-height:1.15;">{model}</div>
                  <div style="font-size:12.5px;color:#94a3b8;margin-top:3px;">{variant}</div>
                </div>
                <div style="background:{b_bg};border:1px solid {b_bd};
                            padding:9px 16px;border-radius:50px;
                            font-size:15px;font-weight:700;color:{b_tx};
                            white-space:nowrap;flex-shrink:0;margin-left:16px;">
                    {score:.0%} match</div>
              </div>

              <!-- spec pills -->
              <div style="display:grid;grid-template-columns:repeat(4,1fr);
                          gap:10px;margin-bottom:18px;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;
                            border-radius:10px;padding:11px 10px;text-align:center;">
                  <div style="font-size:14.5px;font-weight:700;color:#0f172a;">
                      ₹{price:,.0f}</div>
                  <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:0.5px;font-weight:600;margin-top:3px;">
                      Price</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;
                            border-radius:10px;padding:11px 10px;text-align:center;">
                  <div style="font-size:14.5px;font-weight:700;color:#0f172a;">
                      {fuel_t}</div>
                  <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:0.5px;font-weight:600;margin-top:3px;">
                      Fuel</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;
                            border-radius:10px;padding:11px 10px;text-align:center;">
                  <div style="font-size:14.5px;font-weight:700;color:#0f172a;">
                      {mileage:.1f} kmpl</div>
                  <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:0.5px;font-weight:600;margin-top:3px;">
                      Mileage</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;
                            border-radius:10px;padding:11px 10px;text-align:center;">
                  <div style="font-size:14.5px;font-weight:700;color:#0f172a;">
                      {power} BHP</div>
                  <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                              letter-spacing:0.5px;font-weight:600;margin-top:3px;">
                      Power</div>
                </div>
              </div>

              <!-- why box -->
              <div style="background:#f0f7ff;border-left:3px solid #2563eb;
                          border-radius:0 8px 8px 0;padding:11px 15px;
                          font-size:13px;color:#1e3a5f;line-height:1.6;">
                  <strong style="color:#1d4ed8;">Why this car?</strong>
                  &nbsp;{why}
              </div>

            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:32px;padding-top:20px;
            border-top:1px solid #e2e8f0;
            font-size:12px;color:#94a3b8;font-family:Inter,sans-serif;">
    AutoMatch &nbsp;·&nbsp; Intelligent Vehicle Recommender
    &nbsp;·&nbsp; Random Forest &amp; Cosine Similarity
</div>
""", unsafe_allow_html=True)