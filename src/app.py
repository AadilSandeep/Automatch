import streamlit as st
import pandas as pd
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import VehicleRecommender

st.set_page_config(
    page_title="AutoMatch - Intelligent Vehicle Recommender",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    h1 {
        color: #2c3e50;
    }
    h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vehicle_recommender():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "cars.csv")
        return VehicleRecommender(data_path)
    except Exception as e:
        st.error(f"Failed to initialize recommender: {e}")
        return None

recommender = get_vehicle_recommender()


st.markdown("### Intelligent recommendations tailored to your budget and lifestyle.")

col1, col2 = st.columns([1, 2])

with col1:
    with st.container(border=True):
        st.header("Your Preferences")
        
        budget = st.selectbox(
            "Expected Budget Range",
            ["Low", "Mid", "High"],
            help="Low: Affordable, Mid: Balanced, High: Premium" 
        )
        
        fuel = st.selectbox("Preferred Fuel Type", ["Petrol", "Diesel", "CNG"])
        body = st.selectbox("Preferred Body Type", ["SUV", "Sedan", "Hatchback"])
        
        st.subheader("Lifestyle & Usage")
        
        usage = st.selectbox(
            "Primary Usage",
            ["City Commute", "Highway/Long Distance", "Family Trips", "Off-roading/Rough Terrain", "Mixed"]
        )
        
        passengers = st.select_slider(
            "Typical Passengers",
            options=["1-2", "3-4", "5+"]
        )
        
        experience = st.radio(
            "Driving Experience",
            ["First-time Buyer", "Experienced"]
        )
        
        priority = st.selectbox(
            "Top Priority",
            ["Balanced", "Fuel Efficiency", "Performance", "Safety"]
        )
        
        find_btn = st.button("Find My Car 🔍")

    if find_btn and recommender:
        lifestyle_inputs = {
            "usage": usage,
            "passengers": passengers,
            "experience": experience,
            "priority": priority
        }
        
        inferred_specs = recommender.map_lifestyle_to_specs(lifestyle_inputs)
        
        user_prefs = {
            "budget": budget,
            "fuel": fuel,
            "body": body,
            "mileage": inferred_specs.get('mileage', 15.0),
            "seats": inferred_specs.get('seats', 5),
            "power": inferred_specs.get('power', 100.0)
        }
        
     
        
        with st.spinner("Analyzing your lifestyle needs..."):
            recommendations = recommender.recommend(user_prefs)
            st.session_state['recommendations'] = recommendations
            st.session_state['user_prefs'] = user_prefs

with col2:
    if 'recommendations' in st.session_state:
        recs = st.session_state['recommendations']
        prefs = st.session_state.get('user_prefs', {})
        
        if recs.empty:
            st.warning("No vehicles found matching your specific criteria. Try relaxing your constraints.")
        else:
            st.subheader(f"Top {len(recs)} Recommendations")
            
            for i, row in recs.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {row['Make']} {row['Model']}")
                        st.caption(f"{row.get('Variant', '')}")
                    with c2:
                        score = row['Similarity_Score']
                        st.metric("Match Score", f"{score:.0%}")
                    
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.write(f"💰 **₹ {row['Ex-Showroom_Price']:,.0f}**")
                    dc2.write(f"⛽ {row['Fuel_Type']}")
                    dc3.write(f"📏 {row['Mileage']:.1f} kmpl")
                    dc4.write(f"🐎 {row['Power']} BHP")

                    
                    if hasattr(recommender, 'get_explanation'):
                        explanation = recommender.get_explanation(row, prefs)
                        st.info(f"💡 **Why this car?** {explanation}")
                    else:
                        st.info(f"💡 **Why this car?** Matches your {prefs.get('body')} preference.")
