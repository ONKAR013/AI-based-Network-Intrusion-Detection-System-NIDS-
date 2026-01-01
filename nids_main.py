import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze network traffic.
It classifies traffic as:
- **Benign** (Normal Traffic)
- **Malicious** (DDoS / Attack Traffic)
""")

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

df = load_data()

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------
st.sidebar.header("Control Panel")

split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

# 🔍 DEBUG OPTION (HIDDEN BY DEFAULT)
if st.sidebar.checkbox("Show Dataset Columns (Debug)"):
    st.subheader("Dataset Columns")
    st.write(df.columns.tolist())

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
df.columns = df.columns.str.strip()

# Detect target column safely
if "Label" in df.columns:
    target_col = "Label"
elif "label" in df.columns:
    target_col = "label"
elif "Class" in df.columns:
    target_col = "Class"
else:
    target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Limit features for stability (important for CIC-IDS)
X = X.iloc[:, :10]

# Convert labels to numeric
y = pd.to_numeric(y, errors="coerce")
y = y.fillna(0).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=(100 - split_size) / 100,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
st.divider()
col_train, col_metrics = st.columns([1, 2])

with col_train:
    st.subheader("1. Model Training")
    if st.button("Train Model Now"):
        with st.spinner("Training Random Forest Model..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train, y_train)
            st.session_state["model"] = model
        st.success("Model Training Completed!")

# ---------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------
with col_metrics:
    st.subheader("2. Performance Metrics")

    if "model" in st.session_state:
        model = st.session_state["model"]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc * 100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Detected Attacks", int(np.sum(y_pred)))

        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.warning("Please train the model first.")

# ---------------------------------------------------
# LIVE ATTACK SIMULATOR
# ---------------------------------------------------
st.divider()
st.subheader("3. Live Traffic Simulator")

c1, c2, c3, c4 = st.columns(4)

p_dur = c1.number_input("Flow Duration", 0, 100000, 500)
p_pkts = c2.number_input("Total Packets", 0, 500, 100)
p_len = c3.number_input("Packet Length Mean", 0, 1500, 500)
p_active = c4.number_input("Active Mean", 0, 1000, 50)

if st.button("Analyze Packet"):
    if "model" in st.session_state:
        model = st.session_state["model"]

        input_data = np.array(
            [[80, p_dur, p_pkts, p_len, p_active] + [0] * (X.shape[1] - 5)]
        )
        pred = model.predict(input_data)

        if pred[0] == 1:
            st.error("🚨 MALICIOUS TRAFFIC DETECTED")
            st.write("Reason: Abnormal packet behavior detected.")
        else:
            st.success("✅ BENIGN TRAFFIC (Safe)")
    else:
        st.error("Train the model first.")
