import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os, time
from typing import List

# ---------------- Page config ----------------
st.set_page_config(page_title="Phone Price Prediction", page_icon="📱", layout="wide")

# ---------------- Paths ----------------
MODEL_PATH = "Phone_price_prediction.pkl"
TRAIN_CSV_PATH = "22222.csv"   # optional (for nicer dropdowns)

# ---------------- Loaders ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_training_data():
    if os.path.exists(TRAIN_CSV_PATH):
        try:
            return pd.read_csv(TRAIN_CSV_PATH)
        except Exception as e:
            st.warning(f"Could not read {TRAIN_CSV_PATH}: {e}")
    return None

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if "brand" in df.columns:
        df = df.drop(columns=["brand"])
    if "model" in df.columns:
        df = df.drop(columns=["model"])
    if "company of the phone" in df.columns:
        df = df.rename(columns={"company of the phone": "company_phone"})
    return df

# ---------------- Boot ----------------
model = load_model()
df_train = load_training_data()
if df_train is not None:
    df_train = clean_column_names(df_train)

# exact schema expected by model
if hasattr(model, "feature_names_in_"):
    MODEL_FEATURES: List[str] = list(model.feature_names_in_)
else:
    if df_train is None:
        st.error("Model lacks feature_names_in_ and training CSV missing. Cannot determine schema.")
        st.stop()
    cat_cols = [c for c in df_train.columns if c != "price" and (df_train[c].dtype == "object" or str(df_train[c].dtype).startswith("category"))]
    df_proc = pd.get_dummies(df_train, columns=cat_cols)
    if "price" not in df_proc.columns:
        st.error("Target column 'price' not found in training CSV after preprocessing.")
        st.stop()
    MODEL_FEATURES = [c for c in df_proc.columns if c != "price"]

# ---------------- Sidebar (Inputs) ----------------
st.sidebar.image(
    "https://i.pinimg.com/originals/03/c0/38/03c038d16263b1edbd846b4d2cc4ed28.gif ",
    caption="Phone Price Prediction",
    use_column_width=True,
)
st.sidebar.header("Select features to predict phone price")

TRAIN_CHOICES = {}
RAW_COLS = []
if df_train is not None:
    RAW_COLS = [c for c in df_train.columns if c != "price"]
    for c in RAW_COLS:
        if df_train[c].dtype == "object" or str(df_train[c].dtype).startswith("category"):
            TRAIN_CHOICES[c] = sorted(df_train[c].dropna().astype(str).unique().tolist())
else:
    bases = set()
    for f in MODEL_FEATURES:
        bases.add(f.split("_", 1)[0] if "_" in f else f)
    RAW_COLS = sorted(list(bases))

PRIORITY_ORDER = [
    "company_phone", "processor", "Screen_resolution", "color",
    "ram", "display", "storage", "battery_power", "Selfi_Camera", "Main_Camera",
    "rating", "No_of_sim", "fast_charging"
]
ordered_cols = [c for c in PRIORITY_ORDER if c in RAW_COLS] + [c for c in RAW_COLS if c not in PRIORITY_ORDER]

raw_inputs = {}

# --------- Realistic fixed slider ranges for numerics ----------
def realistic_range(col: str):
    lc = col.lower()
    if "ram" in lc:           return 1.0, 16.0, 8.0, 1.0
    if "storage" in lc:       return 8.0, 512.0, 128.0, 8.0
    if "battery" in lc:       return 1000.0, 7000.0, 5000.0, 50.0
    if "display" in lc:       return 4.0, 7.5, 6.5, 0.1
    if "rating" in lc:        return 1.0, 5.0, 4.2, 0.1
    if "sim" in lc:           return 1.0, 2.0, 2.0, 1.0
    if "camera" in lc:        return 5.0, 200.0, 16.0, 1.0
    if "fast_charging" in lc: return 0.0, 1.0, 1.0, 1.0
    return 0.0, 100.0, 50.0, 1.0

def slider_numeric(label: str, key: str, default: float, minv: float, maxv: float, step: float):
    if minv >= maxv:
        minv, maxv = default - step*10, default + step*10
    return st.sidebar.slider(label, min_value=float(minv), max_value=float(maxv), value=float(default), step=float(step), key=key)

for col in ordered_cols:
    label = col.replace("_", " ").title()
    is_categorical_like = (col in TRAIN_CHOICES) or any(f.startswith(col + "_") for f in MODEL_FEATURES)
    if is_categorical_like:
        opts = TRAIN_CHOICES.get(col, [])
        sel = st.sidebar.selectbox(label, options=opts) if opts else st.sidebar.text_input(label)
        raw_inputs[col] = sel
    else:
        mn, mx, mean, step = realistic_range(col)
        val = slider_numeric(label, key=f"num_{col}", default=float(mean), minv=float(mn), maxv=float(mx), step=float(step))
        if float(step).is_integer():
            try: val = int(val)
            except: pass
        raw_inputs[col] = val

# ---------------- Main content (Design) ----------------
st.title("📱 Phone Price Prediction using Machine Learning")
st.image(
    "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?q=80&w=1600&auto=format&fit=crop",
    use_column_width=True,
)
st.subheader("About")
st.write(
    """
This app estimates smartphone prices from specs like RAM, display, storage, battery, cameras, color, brand and processor.  
We align your inputs to the model’s exact **feature schema** (`feature_names_in_`) so predictions never break due to column mismatches.  
Use the sidebar to set specs, then press **Predict Price**. The prediction will run after a short 5-second countdown.
"""
)

# -------- Feature building as model expects --------
def preprocess_single_row(raw_inputs: dict) -> pd.DataFrame:
    row = pd.DataFrame([raw_inputs])
    row = clean_column_names(row)

    dummy_bases = set()
    for c in row.columns:
        if any(f.startswith(c + "_") for f in MODEL_FEATURES):
            dummy_bases.add(c)

    direct_numeric = [f for f in MODEL_FEATURES if f in row.columns]
    for c in direct_numeric:
        row[c] = pd.to_numeric(row[c], errors="coerce")

    row_proc = pd.get_dummies(row, columns=list(dummy_bases)) if dummy_bases else row.copy()

    X = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES, dtype=float)
    for c in row_proc.columns:
        if c in X.columns:
            try:
                X.loc[0, c] = pd.to_numeric(row_proc[c], errors="coerce").values[0]
            except:
                pass
    return X.fillna(0)

# -------------- Predict button (outside sidebar) --------------
predict_clicked = st.button("⏳ Predict Price (runs in 5s)", use_container_width=True)

if predict_clicked:
    # 5-second countdown UI
    info = st.info("Starting prediction in 5 seconds…")
    countdown_ph = st.empty()
    for i in range(5, 0, -1):
        countdown_ph.markdown(f"**{i}**")
        time.sleep(1)
    countdown_ph.empty()
    info.empty()

    try:
        with st.spinner("Predicting…"):
            X_row = preprocess_single_row(raw_inputs)
            pred = model.predict(X_row)[0]
        st.success(f"Estimated Price: **{pred:,.0f}** (dataset currency)")
        with st.expander("Show aligned feature row (debug)"):
            st.dataframe(X_row)
        with st.expander("Model feature names"):
            st.write(f"Total features: {len(MODEL_FEATURES)}")
            st.write(MODEL_FEATURES)
    except Exception as e:
        st.error(f"Prediction failed: {e}")