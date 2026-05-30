# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import sys
import os
import textwrap

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
st.markdown(textwrap.dedent("""
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
"""), unsafe_allow_html=True)


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
    st.markdown(textwrap.dedent("""
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
"""), unsafe_allow_html=True)

    budget = st.selectbox("Budget Range",  ["Low", "Mid", "High"],
                           help="Low ≈ under ₹8L · Mid ≈ ₹8–18L · High ≈ above ₹18L")
    fuel   = st.selectbox("Fuel Type",    ["Petrol", "Diesel", "CNG"])
    body   = st.selectbox("Body Type",    ["SUV", "Sedan", "Hatchback"])

    st.markdown(textwrap.dedent("""
<div style="border-top:1px solid rgba(255,255,255,0.07);margin:20px 0 18px;"></div>
<div style="font-size:10px;font-weight:700;letter-spacing:1.6px;color:#334155;
text-transform:uppercase;margin-bottom:14px;">Lifestyle & Usage</div>
"""), unsafe_allow_html=True)

    usage = st.selectbox(
        "Primary Use",
        ["City Commute", "Highway/Long Distance",
         "Family Trips", "Off-roading/Rough Terrain", "Mixed"],
    )
    passengers = st.select_slider("Passengers Typically", options=["1-2", "3-4", "5+"])
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
        "body":    specs.get("body", body),
        "mileage": specs.get("mileage", 15.0),
        "seats":   specs.get("seats",   5),
        "power":   specs.get("power",   100.0),
        "priority_label": priority,
        "usage_label": usage,
        "experience_label": experience
    }
    with st.spinner("Analysing your preferences…"):
        recs = recommender.recommend(prefs)
        st.session_state["recs"]   = recs
        st.session_state["prefs"]  = prefs
        # Reset compare list when new search happens
        st.session_state["compare_list"] = []


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Results
# ════════════════════════════════════════════════════════════════════════════════

# ── Page header ──────────────────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<div style="margin-bottom:28px;">
<h1 style="font-size:27px;font-weight:800;color:#0f172a;
letter-spacing:-0.6px;margin:0 0 4px;">
Find Your Perfect Car</h1>
<p style="font-size:14px;color:#64748b;margin:0;font-weight:400;">
Recommendations powered by machine learning, personalised to your lifestyle.</p>
</div>
"""), unsafe_allow_html=True)

# ── Stats row ────────────────────────────────────────────────────────────────────
# Removed at user request to simplify UI


# ── Empty state ───────────────────────────────────────────────────────────────────
if "recs" not in st.session_state:
    st.markdown(textwrap.dedent("""
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
"""), unsafe_allow_html=True)

else:
    recs  = st.session_state["recs"]
    prefs = st.session_state["prefs"]
    
    # Initialize comparison list
    if "compare_list" not in st.session_state:
        st.session_state["compare_list"] = []

    # ── User Preference Summary ──────────────────────────────────────────────────────
    st.markdown(textwrap.dedent(f"""
<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:16px 20px; margin-bottom:24px; display:flex; align-items:center; gap:12px;">
<div style="font-size:24px;">👤</div>
<div>
<div style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Your Profile</div>
<div style="font-size:14px; color:#334155; margin-top:2px;">
Looking for a <strong>{prefs['budget']}-budget {prefs['fuel']} {prefs['body']}</strong> for <strong>{prefs['usage_label']}</strong>, prioritizing <strong>{prefs['priority_label']}</strong>.
</div>
</div>
</div>
"""), unsafe_allow_html=True)

    if recs.empty:
        st.warning(f"No cars found matching your strict criteria (Budget: {prefs['budget']}). Try relaxing filters.")

    else:
        n = len(recs)
        
        # ── Fallback Message ─────────────────────────────────────────────────────────────
        first_match_type = recs.iloc[0].get('Match_Type', 'Exact Match')
        if first_match_type != 'Exact Match':
            st.info(f"💡 No exact **{prefs['budget']} {prefs['fuel']} {prefs['body']}** found. Here are some **{first_match_type}** alternatives:")
        
        st.markdown(textwrap.dedent(f"""
<div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:18px;">
Results
<span style="background:#eff6ff;color:#2563eb;font-size:12px;
font-weight:600;padding:3px 10px;border-radius:6px;
margin-left:8px;">{n} match{'es' if n != 1 else ''}</span>
</div>
"""), unsafe_allow_html=True)

        for rank, (_, row) in enumerate(recs.iterrows(), start=1):
            score   = row["Similarity_Score"]
            price   = row["Ex-Showroom_Price"]
            mileage = row["Mileage"]
            power   = row["Power"]
            fuel_t  = row["Fuel_Type"]
            make    = row["Make"]
            model   = row["Model"]
            variant = row.get("Variant", "")
            badges  = row.get("Badges", [])
            
            # Scores breakdown
            b_match = row.get("Budget_Match", 1.0) * 100
            f_match = row.get("Fuel_Match", 1.0) * 100
            m_match = row.get("Mileage_Match", 0) * 100
            l_match = row.get("Lifestyle_Match", 0) * 100

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

            # badge pills HTML
            badges_html = ""
            for b in badges:
                badges_html += f'<span style="background:#fef3c7; border:1px solid #fde68a; color:#b45309; font-size:10px; font-weight:700; padding:3px 8px; border-radius:4px; margin-left:8px; text-transform:uppercase;">{b}</span>'

            # A simple brand initial/logo placeholder
            brand_initial = str(make)[0].upper() if pd.notna(make) and str(make).strip() else "🚗"

            st.markdown(textwrap.dedent(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;
padding:26px 28px;margin-bottom:8px;
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
<div style="display:flex; gap:16px; align-items:center;">
<div style="width:56px; height:56px; background:#f1f5f9; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px; font-weight:800; color:#cbd5e1; border:1px solid #e2e8f0;">
{brand_initial}
</div>
<div>
<div style="font-size:11px;font-weight:700;letter-spacing:1.2px;
text-transform:uppercase;color:#2563eb;margin-bottom:2px; display:flex; align-items:center;">
{make} {badges_html}
</div>
<div style="font-size:22px;font-weight:800;color:#0f172a;
letter-spacing:-0.4px;line-height:1.15;">{model}</div>
<div style="font-size:12.5px;color:#94a3b8;margin-top:3px;">{variant}</div>
</div>
</div>
<div style="text-align:right;">
<div style="background:{b_bg};border:1px solid {b_bd};
padding:9px 16px;border-radius:50px;
font-size:15px;font-weight:700;color:{b_tx};
white-space:nowrap;flex-shrink:0;margin-bottom:8px;">
{score:.0%} match</div>

<!-- Score Breakdown Mini -->
<div style="display:flex; gap:6px; font-size:10px; color:#64748b; font-weight:600; justify-content:flex-end;">
<span title="Budget Match">💰 {b_match:.0f}%</span>
<span title="Fuel Match">⛽ {f_match:.0f}%</span>
<span title="Mileage Match">🛣️ {m_match:.0f}%</span>
<span title="Lifestyle Match">⚙️ {l_match:.0f}%</span>
</div>
</div>
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
"""), unsafe_allow_html=True)
            
            # Compare Checkbox right below the card
            col1, col2 = st.columns([1, 4])
            with col1:
                # Use a session state to track if selected
                is_selected = f"{make} {model}" in st.session_state["compare_list"]
                if st.checkbox(f"Compare #{rank}", value=is_selected, key=f"comp_{rank}_{make}_{model}"):
                    if f"{make} {model}" not in st.session_state["compare_list"]:
                        st.session_state["compare_list"].append(f"{make} {model}")
                else:
                    if f"{make} {model}" in st.session_state["compare_list"]:
                        st.session_state["compare_list"].remove(f"{make} {model}")
                        
            st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

        # ── Comparison Section ──────────────────────────────────────────────────────────
        if len(st.session_state["compare_list"]) > 0:
            st.markdown(textwrap.dedent("""
<div style="margin-top:40px; border-top:1px solid #e2e8f0; padding-top:24px;">
<h3 style="font-size:20px; font-weight:800; color:#0f172a; margin-bottom:16px;">Car Comparison</h3>
</div>
"""), unsafe_allow_html=True)
            
            # Filter the recs for only selected cars
            # Create a full name column for filtering
            recs_copy = recs.copy()
            recs_copy['FullName'] = recs_copy['Make'] + " " + recs_copy['Model']
            
            compare_df = recs_copy[recs_copy['FullName'].isin(st.session_state["compare_list"])]
            
            if not compare_df.empty:
                # Transpose for easier side-by-side comparison
                cols_to_compare = [
                    "FullName", "Ex-Showroom_Price", "Fuel_Type", "Mileage", "Seating_Capacity", 
                    "Power", "Torque", "Displacement", "Gears", "Drivetrain", 
                    "Ground_Clearance", "Number_of_Airbags", "Boot_Space", "Wheelbase", 
                    "Body_Type", "Similarity_Score"
                ]
                
                # Make sure all columns exist in the dataframe to prevent KeyError
                cols_to_compare = [col for col in cols_to_compare if col in compare_df.columns]
                
                disp_df = compare_df[cols_to_compare].set_index("FullName").T
                
                # Define row labels map
                row_labels = {
                    "Ex-Showroom_Price": "Price",
                    "Fuel_Type": "Fuel Type",
                    "Mileage": "Mileage",
                    "Seating_Capacity": "Seats",
                    "Power": "Power",
                    "Torque": "Torque",
                    "Displacement": "Engine Displacement",
                    "Gears": "Transmission (Gears)",
                    "Drivetrain": "Drivetrain",
                    "Ground_Clearance": "Ground Clearance",
                    "Number_of_Airbags": "Airbags",
                    "Boot_Space": "Boot Space",
                    "Wheelbase": "Wheelbase",
                    "Body_Type": "Body Type",
                    "Similarity_Score": "Match Score"
                }
                
                # Format functions for each spec
                formatted_rows = {}
                for col_name in cols_to_compare:
                    if col_name == "FullName":
                        continue
                    row_data = disp_df.loc[col_name]
                    formatted_values = []
                    for val in row_data:
                        if pd.isna(val) or str(val).strip().lower() == "nan" or str(val).strip() == "":
                            formatted_values.append("Not Specified" if col_name == "Number_of_Airbags" else "N/A")
                        else:
                            if col_name == "Ex-Showroom_Price":
                                formatted_values.append(f"₹{float(val):,.0f}")
                            elif col_name == "Mileage":
                                formatted_values.append(f"{float(val):.1f} kmpl")
                            elif col_name == "Power":
                                formatted_values.append(f"{int(float(val))} BHP")
                            elif col_name == "Torque":
                                formatted_values.append(f"{int(float(val))} Nm")
                            elif col_name == "Displacement":
                                formatted_values.append(f"{int(float(val))} cc")
                            elif col_name == "Number_of_Airbags":
                                formatted_values.append(f"{int(float(val))} Airbags")
                            elif col_name == "Wheelbase":
                                formatted_values.append(f"{int(float(val))} mm")
                            elif col_name == "Similarity_Score":
                                formatted_values.append(f"{float(val):.0%}")
                            elif col_name == "Gears":
                                if "gear" in str(val).lower():
                                    formatted_values.append(str(val))
                                else:
                                    formatted_values.append(f"{val} Gears")
                            else:
                                formatted_values.append(str(val))
                    formatted_rows[col_name] = formatted_values
                
                # Build HTML table
                car_names = list(disp_df.columns)
                
                # Table style definition
                html = textwrap.dedent("""
                <style>
                .compare-table-container {
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 14px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
                    overflow: hidden;
                    margin-bottom: 24px;
                }
                .compare-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'Inter', sans-serif;
                    font-size: 13.5px;
                }
                .compare-table th {
                    background: linear-gradient(175deg, #0c1628 0%, #0f1f3d 55%, #0d2847 100%);
                    color: #ffffff;
                    text-align: left;
                    font-weight: 700;
                    padding: 14px 18px;
                    border-bottom: 1px solid rgba(255,255,255,0.06);
                }
                .compare-table th:first-child {
                    width: 22%;
                    font-weight: 800;
                    letter-spacing: 0.5px;
                }
                .compare-table td {
                    padding: 12px 18px;
                    border-bottom: 1px solid #edf2f7;
                    color: #334155;
                }
                .compare-table tr:last-child td {
                    border-bottom: none;
                }
                /* Alternating rows */
                .compare-table tr:nth-child(even) {
                    background-color: #f8fafc;
                }
                /* Hover effect */
                .compare-table tr:hover {
                    background-color: #f1f5f9;
                    transition: background-color 0.15s ease-in-out;
                }
                /* Spec label column */
                .compare-table td.spec-label {
                    font-weight: 600;
                    color: #475569;
                    border-right: 1px solid #e2e8f0;
                }
                /* Match Score Row Highlight */
                .compare-table tr.match-score-row {
                    background: linear-gradient(90deg, #ecfdf5 0%, #f0fdf4 100%) !important;
                    font-weight: 700;
                }
                .compare-table tr.match-score-row td {
                    color: #15803d !important;
                }
                </style>
                """)
                
                html += '<div class="compare-table-container">'
                html += '<table class="compare-table">'
                
                # Headers (Car Names)
                html += '<thead><tr><th>Feature</th>'
                for name in car_names:
                    html += f'<th>{name}</th>'
                html += '</tr></thead><tbody>'
                
                # Body Rows
                for col_name in cols_to_compare:
                    if col_name == "FullName":
                        continue
                    
                    label = row_labels.get(col_name, col_name)
                    values = formatted_rows[col_name]
                    
                    is_match_score = (col_name == "Similarity_Score")
                    row_class = ' class="match-score-row"' if is_match_score else ''
                    
                    html += f'<tr{row_class}>'
                    html += f'<td class="spec-label">{label}</td>'
                    for val in values:
                        html += f'<td>{val}</td>'
                    html += '</tr>'
                
                html += '</tbody></table></div>'
                
                st.markdown(html, unsafe_allow_html=True)
                
            if len(st.session_state["compare_list"]) > 3:
                st.warning("You selected more than 3 cars. It might be hard to read.")

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown(textwrap.dedent("""
<div style="text-align:center;margin-top:32px;padding-top:20px;
border-top:1px solid #e2e8f0;
font-size:12px;color:#94a3b8;font-family:Inter,sans-serif;">
AutoMatch &nbsp;·&nbsp; Intelligent Vehicle Recommender
&nbsp;·&nbsp; Random Forest &amp; Cosine Similarity
</div>
"""), unsafe_allow_html=True)