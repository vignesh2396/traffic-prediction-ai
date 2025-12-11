# Traffic Congestion Prediction & Smart Agent System

This project presents an AI-powered traffic congestion prediction system integrated with a rule-based traffic agent. Built using **Python, Streamlit, scikit-learn**, and **Google Gemini API**.

---

## 🚦 Features

- Predicts traffic congestion using trained ML models.
- Real-time user inputs: junction, time, vehicle count, weather.
- Suggests traffic control decisions via a rule-based agent.
- Utilizes Google Gemini for explanation and summarization.
- Streamlit-powered user interface.

---
## What this app does

- **Prediction:** Loads pre-trained artifacts (model, scaler, label encoders) and predicts Low/Medium/High congestion for a selected junction and conditions.
- **Agent suggestion:** Calls Google Gemini with a structured prompt to produce a short recommendation (signal timing adjustments, rerouting, alerts).
- **Interactive UI:** Streamlit sidebar for inputs; results shown inline with agent output.

---

## Project structure

```
traffic-agent/
├─ app.py                        # The Streamlit app (your code)
├─ requirements.txt              # Python dependencies
├─ traffic_model.pkl             # Trained model (e.g., XGBoost/GBM)
├─ traffic_scaler.pkl            # Preprocessing scaler
├─ traffic_label_encoders.pkl    # Dict with encoders for Date, Time, Weather
└─ .streamlit/
   └─ secrets.toml               # Optional: Gemini API key storage for Streamlit
```

> Note: Ensure the three .pkl files exist and match the training pipeline (same feature order and encoders).

---

## Prerequisites

- **Python:** 3.9–3.11 recommended.
- **Google Gemini API key:** Create in Google AI Studio and copy the key.
- **Model artifacts:** `traffic_model.pkl`, `traffic_scaler.pkl`, `traffic_label_encoders.pkl`.

---

## Create and activate a virtual environment (Windows)

1. **Create venv**
   ```bash
   python -m venv .venv
   ```

2. **Activate venv**
   ```bash
   .venv\Scripts\activate
   ```

3. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

> macOS/Linux activation:
> - bash/zsh: `source .venv/bin/activate`

---

## Install dependencies

- **Create requirements.txt**
  ```text
  streamlit==1.38.0
  numpy==1.26.4
  scikit-learn==1.3.2
  xgboost==1.7.6
  google-generativeai==0.7.2
  ```
- **Install**
  ```bash
  pip install -r requirements.txt
  ```

---

## Add your Gemini API key

Choose one of the following methods.

### Option A — Environment variable
- **Set the variable (PowerShell):**
  ```powershell
  $Env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
  ```
- **Set the variable (cmd):**
  ```cmd
  set GEMINI_API_KEY=YOUR_API_KEY_HERE
  ```

### Option B — Streamlit secrets
- **Create file:** `.streamlit/secrets.toml`
  ```toml
  GEMINI_API_KEY = "YOUR_API_KEY_HERE"
  ```
- The app automatically checks Streamlit secrets first, then environment variables.

---

## Run the app

```bash
streamlit run app.py
```

- Open the local URL Streamlit prints (usually http://localhost:8501).
- Use the sidebar to set junction, date, time, vehicle count, speed, weather, green/red times.
- Click “Ask Agent” to get Gemini’s recommendation.

---

## How the code works

- **Page setup:** Configures Streamlit page and title.
- **Load artifacts:** Reads `traffic_model.pkl` (predictor), `traffic_scaler.pkl` (feature scaling), and `traffic_label_encoders.pkl` (categorical encoders for Date/Time/Weather).
- **Inputs:** Sidebar controls collect traffic context: junction, date/time, vehicle count, speed, weather, signal durations.
- **Encoding & scaling:** Applies label encoders to categorical fields and scales the final feature vector with the saved scaler.
- **Prediction:** Uses the loaded model to predict a class mapped to Low/Medium/High congestion.
- **Gemini setup:** Retrieves `GEMINI_API_KEY` from Streamlit secrets or environment; configures `google.generativeai`.
- **Agent call:** Builds a concise prompt with the current scenario and calls Gemini. The helper function tries multiple client methods to support different `google-generativeai` versions.
- **Response parsing:** Extracts text from common response shapes and displays it in the UI.
- **Error handling:** Shows warnings when API key is missing and readable error messages if the client call fails.

---

## Tips and troubleshooting

- **Model artifacts mismatch:** If you see encoder or scaler errors, ensure training and inference feature orders match:
  - `[junction, date_encoded, time_encoded, vehicle_count, speed, weather_encoded, green, red]`
- **Gemini model name:** The code uses `gemini-2.5-flash`. If unavailable in your region/version, try `gemini-1.5-flash`.
- **API client versions:** The helper probes multiple methods (`get_model`, `GenerativeModel`, `models.TextGenerationModel`). Keep `google-generativeai` up to date if calls fail.
  ```bash
  pip install --upgrade google-generativeai
  ```
- **Environment not detected:** Activate the venv before running `streamlit run app.py`.
- **Secrets not picked up:** Confirm `.streamlit/secrets.toml` is in the project root and formatted as TOML.

---

## 🚦 Sample input ranges for congestion levels

| Condition | Vehicle Count | Avg Speed (km/h) | Signal (Green/Red sec) | Weather | Expected Congestion |
|-----------|---------------|------------------|------------------------|---------|---------------------|
| **Light traffic** | 0–40 | 70–120 | Green ≥ 60, Red ≤ 30 | Clear | **Low (0)** |
| **Moderate traffic** | 40–100 | 40–70 | Balanced signals (Green ~40–60, Red ~40–60) | Cloudy/Light rain | **Medium (1)** |
| **Heavy traffic** | 100–200 | 0–40 | Green ≤ 30, Red ≥ 60 | Rain/Storm | **High (2)** |
| **Unknown** | Any values outside trained encoder classes (e.g., invalid Date/Time/Weather not in `traffic_label_encoders.pkl`) | — | — | — | **Unknown** |

---

## 🔑 How this maps to your code

- **Vehicle Count & Speed** are the strongest drivers:
  - Low congestion → fewer vehicles, higher speeds.
  - High congestion → many vehicles, slower speeds.
- **Signal timings** influence flow:
  - Longer green reduces congestion.
  - Longer red increases congestion.
- **Weather** adds context:
  - Clear → more likely Low/Medium.
  - Rain/Storm → more likely Medium/High.
- **Unknown** happens if:
  - The model predicts a class not in `{0,1,2}`.
  - Or encoders fail (e.g., you input a Date/Weather not seen during training).

---
## License

- **Usage:** Educational prototype for traffic agenting. Adapt and extend for your context.
