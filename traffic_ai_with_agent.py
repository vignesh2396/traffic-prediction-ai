import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import google.generativeai as genai
import joblib

# Page config
st.set_page_config(page_title="Traffic Congestion Predictor & AI Agent", layout="wide")
st.title("🚦 Prediction & Agent Simulation")
st.markdown("This app predicts congestion and lets an agent take action, assisted by **Google Gemini**.")

# === LOAD MODEL & SCALER ===
model = joblib.load("traffic_model.pkl")

scaler_bundle = joblib.load("traffic_scaler.pkl")
scaler = scaler_bundle["scaler"]
features = scaler_bundle["features"]

encoders = joblib.load("traffic_label_encoders.pkl")
le_date = encoders["Date"]
le_time = encoders["Time"]
le_weather = encoders["Weather"]

# === SIDEBAR USER INPUTS ===
st.sidebar.header("Input Traffic Data")

# ✅ Use encoded form directly (e.g., 0 for J1, 1 for J2)
junction = st.sidebar.selectbox("Junction ID", [0, 1], format_func=lambda x: f"J{x+1}") or 0
date = st.sidebar.selectbox("Date", le_date.classes_.tolist())
time = st.sidebar.selectbox("Time", le_time.classes_.tolist())
vehicle_count = st.sidebar.slider("Vehicle Count", 0, 200, 50)
speed = st.sidebar.slider("Average Speed (km/h)", 0, 120, 45)
weather = st.sidebar.selectbox("Weather", le_weather.classes_.tolist())
green = st.sidebar.slider("Signal State - Green (sec)", 0, 120, 50)
red = st.sidebar.slider("Signal State - Red (sec)", 0, 120, 50)

# === ENCODE CATEGORICAL FEATURES ===
date_encoded = le_date.transform([date])[0]
time_encoded = le_time.transform([time])[0]
weather_encoded = le_weather.transform([weather])[0]

# === PREPARE FINAL INPUT ===
# input_features = np.array([[junction, date_encoded, time_encoded, vehicle_count,
#                             speed, weather_encoded, green, red]])
# input_scaled = scaler.transform(input_features)

# 1) Read the training-time feature names (and order) from the scaler
try:
    train_cols = scaler.feature_names_in_.tolist()
except AttributeError:
    # If your scaler was fit without feature names, fall back to numpy (no names) to avoid errors
    train_cols = None

# 2) Map your UI variables to the training column names
# Adjust these names to match exactly what you used during training.
# From your error message, these appear to be:
# - Average_Speed_kmph
# - Date
# - Junction_ID
# - Signal_State_Green
# - Signal_State_Red
# - Time
# - Vehicle_Count
# - Weather
row_as_training_names = {
    "Junction_ID": junction,                # encoded int (0 for J1, 1 for J2)
    "Date": date_encoded,                   # label-encoded date
    "Time": time_encoded,                   # label-encoded time
    "Vehicle_Count": vehicle_count,
    "Average_Speed_kmph": speed,
    "Weather": weather_encoded,
    "Signal_State_Green": green,
    "Signal_State_Red": red,
}

# 3) Build the DataFrame with the exact columns in the exact order used during fit
if train_cols:
    # Ensure all required columns are present
    missing = [c for c in train_cols if c not in row_as_training_names]
    extra = [c for c in row_as_training_names if c not in train_cols]
    if missing:
        st.error(f"Missing features for scaler: {missing}. Check your training column names.")
    if extra:
        st.warning(f"Extra features not used by scaler: {extra}. They will be ignored.")
    # Order strictly by training columns
    input_df = pd.DataFrame([[row_as_training_names[c] for c in train_cols]], columns=train_cols)
    input_scaled = scaler.transform(input_df)
else:
    # Fallback: use numpy array in the same order you trained the scaler
    # Update this order to match your training pipeline exactly.
    input_features = np.array([[
        row_as_training_names["Junction_ID"],
        row_as_training_names["Date"],
        row_as_training_names["Time"],
        row_as_training_names["Vehicle_Count"],
        row_as_training_names["Average_Speed_kmph"],
        row_as_training_names["Weather"],
        row_as_training_names["Signal_State_Green"],
        row_as_training_names["Signal_State_Red"],
    ]])
    input_scaled = scaler.transform(input_features)

# === PREDICT ===
prediction = model.predict(input_scaled)[0]
congestion_map = {0: "Low", 1: "Medium", 2: "High"}
congestion_level = congestion_map.get(prediction, "Unknown")

st.subheader("🚗 Predicted Congestion Level")
st.success(f"**{congestion_level}** congestion expected at Junction {junction + 1}")

# === GEMINI AGENT INTERACTION ===
st.subheader("🤖 Agent Suggestion")
# Google Gemini Setup - configure from environment or Streamlit secrets
api_key = None
if hasattr(st, "secrets") and isinstance(st.secrets, dict):
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
else:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("No Gemini API key found. Set `GEMINI_API_KEY` environment variable or add it to Streamlit secrets.")
else:
    genai.configure(api_key=api_key)

prompt = f"""
You are a traffic management AI agent.

Given the current traffic scenario:
- Junction: J{junction + 1}
- Date: {date}
- Time: {time}
- Vehicle Count: {vehicle_count}
- Speed: {speed} km/h
- Weather: {weather}
- Signal: Green={green}s, Red={red}s
- Predicted Congestion: {congestion_level}

Suggest actions like:
- Adjusting signal timings
- Rerouting traffic
- Alerting nearby junctions
- Triggering emergency responses (if needed)

Provide a short and precise recommendation.
"""

if st.button("💡 Ask Agent"):
    with st.spinner("Consulting agent..."):
        if not api_key:
            st.error("Cannot consult agent: missing API key.")
        else:
            try:
                # Helper: detect available API surface and call it
                def _call_genai(prompt_text, model_name="gemini-2.5-flash"):
                    # 1) Try genai.get_model(...).generate(...) pattern
                    if hasattr(genai, "get_model"):
                        try:
                            m = genai.get_model(model_name)
                            if hasattr(m, "generate"):
                                return m.generate(prompt_text)
                            if hasattr(m, "generate_text"):
                                return m.generate_text(prompt_text)
                            if hasattr(m, "call"):
                                return m.call(prompt_text)
                        except Exception:
                            pass

                    # 2) Try genai.GenerativeModel class patterns
                    if hasattr(genai, "GenerativeModel"):
                        GM = genai.GenerativeModel
                        # try classmethod from_pretrained
                        if hasattr(GM, "from_pretrained"):
                            try:
                                inst = GM.from_pretrained(model_name)
                                if hasattr(inst, "generate_content"):
                                    return inst.generate_content(prompt_text)
                                if hasattr(inst, "generate"):
                                    return inst.generate(prompt_text)
                                if hasattr(inst, "generate_text"):
                                    return inst.generate_text(prompt_text)
                            except Exception:
                                pass

                        # try constructing with model name (some versions accept model as first arg)
                        try:
                            inst = GM(model_name)
                            if hasattr(inst, "generate_content"):
                                return inst.generate_content(prompt_text)
                            if hasattr(inst, "generate"):
                                return inst.generate(prompt_text)
                            if hasattr(inst, "generate_text"):
                                return inst.generate_text(prompt_text)
                        except Exception:
                            pass

                    # 3) Some modules expose a `models` namespace
                    if hasattr(genai, "models"):
                        try:
                            models_ns = genai.models
                            if hasattr(models_ns, "TextGenerationModel"):
                                TextModel = getattr(models_ns, "TextGenerationModel")
                                try:
                                    inst = TextModel.from_pretrained(model_name)
                                    if hasattr(inst, "generate"):
                                        return inst.generate(prompt_text)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # Nothing matched — raise an informative error
                    available = ", ".join([a for a in dir(genai) if not a.startswith("__")])
                    raise RuntimeError(
                        "google.generativeai does not expose a supported generate API in this installation. "
                        f"Available attributes: {available}. Try installing a compatible version (see README)."
                    )

                response = _call_genai(prompt)

                # Extract text from common response shapes
                text = ""
                if response is None:
                    text = "(no response)"
                elif hasattr(response, "text") and response.text:
                    text = response.text
                elif isinstance(response, dict):
                    if "candidates" in response and response["candidates"]:
                        text = response["candidates"][0].get("content", "")
                    elif "output" in response and isinstance(response["output"], list):
                        parts = [str(item.get("content", "")) for item in response["output"]]
                        text = "\n".join([p for p in parts if p])
                    elif "text" in response:
                        text = str(response["text"])
                    else:
                        text = str(response)
                else:
                    # Some client objects implement __str__ or return a sequence
                    try:
                        text = str(response)
                    except Exception:
                        text = "(unreadable response)"

                st.info(text)
            except Exception as e:
                # Provide actionable debugging info for the user
                st.error("Agent error: could not call Gemini generation API.")
                st.caption(
                    "Details: " + str(e) + " — if this persists, check your installed `google-generativeai` package version."
                )

# === FOOTER ===
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, XGBoost, and Gemini AI")
