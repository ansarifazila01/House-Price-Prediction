
import streamlit as st
import pandas as pd
import json
import pickle
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EstateIQ | House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_dataset():
    with open(DATA_DIR / "price.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)

@st.cache_resource
def load_model():
    with open(DATA_DIR / "house_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    df = load_dataset()
    model = load_model()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f3f7f8;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    background: linear-gradient(120deg, #102d3a, #176b70);
    padding: 48px 35px;
    border-radius: 18px;
    color: white;
    min-height: 235px;
}
.hero h1 {
    font-size: 43px;
    color: white;
    margin-bottom: 12px;
}
.hero p {
    color: #d8eeee;
    font-size: 17px;
}
.welcome {
    background: linear-gradient(145deg, #168b87, #0e625f);
    padding: 30px;
    border-radius: 18px;
    color: white;
    min-height: 235px;
}
.welcome h2 {
    color: white;
}
.welcome p {
    color: #e0f5f2;
    font-size: 16px;
}
.section-title {
    color: #153d48;
    font-size: 25px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 15px;
}
.info-card {
    background: white;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 15px;
    border: 1px solid #e1ebec;
}
.info-card h4 {
    color: #176b70;
    margin-bottom: 8px;
}
.info-card p {
    color: #53676c;
    font-size: 14px;
}
.result-card {
    background: #e1f3ef;
    border: 1px solid #a8d9cd;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}
.result-card h2 {
    color: #12665c;
}
div.stButton > button {
    background: #147c78;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 12px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #105e5b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
left, right = st.columns([1.65, 1])

with left:
    st.markdown("""
    <div class="hero">
        <h1>EstateIQ</h1>
        <p>Smart Property Price Prediction</p>
        <p>Make informed property decisions with data-driven price estimates.</p>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="welcome">
        <h2>Welcome!</h2>
        <p>Find an estimated property value by entering your property details.</p>
        <p>Simple. Smart. Data-driven.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PROPERTY DETAILS ----------------
st.markdown('<div class="section-title">Property Details</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([1.25, 1])

with col1:
    st.markdown("### Enter Your Property Information")

    property_types = sorted(df["bedrooms"].dropna().unique().tolist())
    locations = sorted(df["location"].dropna().unique().tolist())

    with st.form("prediction_form"):
        property_type = st.selectbox(
            "Property Type",
            property_types
        )

        location = st.selectbox(
            "Location",
            locations
        )

        area = st.number_input(
            "Area (sq. ft.)",
            min_value=100,
            max_value=20000,
            value=1500,
            step=50
        )

        predict_button = st.form_submit_button(
            "Predict Property Price",
            use_container_width=True
        )

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>📍 Location Matters</h4>
        <p>Property prices can vary depending on the selected location.</p>
    </div>
    <div class="info-card">
        <h4>🏠 Property Type</h4>
        <p>Select the property category available in the dataset.</p>
    </div>
    <div class="info-card">
        <h4>📐 Property Area</h4>
        <p>Enter the total area of your property in square feet.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_button:
    input_data = pd.DataFrame([{
        "bedrooms": property_type,
        "location": location,
        "area": area
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        # Dataset prices are treated as lakh INR
        price_in_lakh = prediction

        st.markdown('<div class="section-title">Estimated Property Value</div>',
                    unsafe_allow_html=True)

        if price_in_lakh >= 100:
            price_text = f"₹{price_in_lakh / 100:.2f} Crore"
        else:
            price_text = f"₹{price_in_lakh:.2f} Lakh"

        st.markdown(f"""
        <div class="result-card">
            <h2>{price_text}</h2>
            <p>Estimated Property Price</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is an approximate prediction based on the Hyderabad property dataset. "
            "Actual market prices may vary."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>EstateIQ | Smart House Price Prediction</center>",
    unsafe_allow_html=True
)
