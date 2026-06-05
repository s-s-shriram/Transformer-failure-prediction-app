import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="GRIDTITAN - Transformer Risk Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GRIDTITAN - Transformer Risk Intelligence System")
st.write(
    "AI prototype for transformer risk prediction, root-cause analysis, "
    "and priority maintenance planning."
)


# ==============================
# HELPER FUNCTIONS
# ==============================
def classify_risk(score):
    if score >= 0.70:
        return "High"
    elif score >= 0.40:
        return "Medium"
    return "Low"


def risk_status(level):
    if level == "High":
        return "🔴 HIGH RISK"
    elif level == "Medium":
        return "🟡 MEDIUM RISK"
    return "🟢 LOW RISK"


def maintenance_action(level):
    if level == "High":
        return "Immediate inspection required"
    elif level == "Medium":
        return "Monitor closely"
    return "Routine monitoring"


def root_cause_from_values(
    load,
    temperature,
    voltage_deviation,
    current_stress,
    load_fluctuation,
    current_stress_threshold=0.20,
    load_fluctuation_threshold=25
):
    if load > 85 and temperature > 40:
        return "Thermal Overload"
    elif voltage_deviation > 15:
        return "Voltage Instability"
    elif current_stress > current_stress_threshold:
        return "Electrical Stress"
    elif load_fluctuation > load_fluctuation_threshold:
        return "Sudden Load Variation"
    return "General Degradation"


def rule_based_score(load, temperature, voltage, current, power):
    voltage_deviation = abs(voltage - 230)

    score = 0
    score += 0.30 if load > 100 else 0.15 if load > 80 else 0
    score += 0.25 if temperature > 45 else 0.10 if temperature > 38 else 0
    score += 0.20 if voltage_deviation > 18 else 0.08 if voltage_deviation > 10 else 0
    score += 0.15 if current > 65 else 0.05 if current > 50 else 0
    score += 0.10 if power > 15 else 0

    return min(score, 0.99)


# ==============================
# SESSION STATE
# ==============================
if "auto_id" not in st.session_state:
    st.session_state.auto_id = 1001

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if "features" not in st.session_state:
    st.session_state.features = None

if "current_stress_threshold" not in st.session_state:
    st.session_state.current_stress_threshold = 0.20

if "load_fluctuation_threshold" not in st.session_state:
    st.session_state.load_fluctuation_threshold = 25


# ==============================
# TRANSFORMER TYPE
# ==============================
transformer_type = st.selectbox(
    "Select Transformer Type",
    ["Distribution Transformer", "Power Transformer", "Pole-Mounted Transformer"],
    index=0
)

st.info(
    f"Current focus: **{transformer_type}**. "
    "For PS17, the primary focus is distribution transformers used in power distribution networks."
)


# ==============================
# TABS
# ==============================
tab1, tab2 = st.tabs(
    ["📂 Upload CSV & Train Model", "🧪 Manual Transformer Test"]
)


# ======================================================
# TAB 1: CSV UPLOAD MODE
# ======================================================
with tab1:

    with st.expander("📄 Expected CSV Format"):
        sample = pd.DataFrame({
            "transformer_id": ["TR-1001", "TR-1002", "TR-1003"],
            "load": [75, 95, 120],
            "temperature": [35, 45, 52],
            "voltage": [230, 238, 248],
            "current": [40, 58, 75],
            "power": [9.2, 13.8, 18.6],
            "failure": [0, 0, 1]
        })

        st.dataframe(sample)

        st.download_button(
            label="⬇️ Download Sample CSV Format",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="sample_transformer_dataset.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(
        "Upload Transformer Dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.subheader("📌 Dataset Preview")
        st.dataframe(data.head())

        required_cols = [
            "transformer_id",
            "load",
            "temperature",
            "voltage",
            "current",
            "power",
            "failure"
        ]

        missing_cols = [
            col for col in required_cols if col not in data.columns
        ]

        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            st.stop()

        st.subheader("🔍 Missing Value Summary")
        st.write(data.isnull().sum())

        # ==============================
        # KNN IMPUTATION
        # ==============================
        numeric_cols = [
            "load",
            "temperature",
            "voltage",
            "current",
            "power"
        ]

        imputer = KNNImputer(n_neighbors=5)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

        # ==============================
        # FEATURE ENGINEERING
        # ==============================
        data["thermal_stress"] = data["load"] * data["temperature"]
        data["overload"] = (data["load"] > 80).astype(int)
        data["load_ratio"] = data["load"] / 100
        data["voltage_deviation"] = abs(data["voltage"] - 230)
        data["current_stress"] = data["current"] / (data["voltage"] + 1)

        data = data.sort_values(by=["transformer_id"])
        data["load_fluctuation"] = (
            data.groupby("transformer_id")["load"]
            .diff()
            .fillna(0)
            .abs()
        )

        features = [
            "load",
            "temperature",
            "voltage",
            "current",
            "power",
            "thermal_stress",
            "overload",
            "load_ratio",
            "voltage_deviation",
            "current_stress",
            "load_fluctuation"
        ]

        X = data[features]
        y = data["failure"]

        class_counts = y.value_counts()

        if len(class_counts) < 2 or class_counts.min() < 2:
            st.error(
                "Dataset must contain enough samples for both failure and non-failure classes."
            )
            st.stop()

        # ==============================
        # SMOTE
        # ==============================
        smote = SMOTE(
            random_state=42,
            k_neighbors=min(5, class_counts.min() - 1)
        )

        X_res, y_res = smote.fit_resample(X, y)

        st.subheader("⚖️ Class Balance After SMOTE")
        st.write(pd.Series(y_res).value_counts())

        # ==============================
        # TRAIN TEST SPLIT
        # ==============================
        X_train, X_test, y_train, y_test = train_test_split(
            X_res,
            y_res,
            test_size=0.2,
            random_state=42
        )

        # ==============================
        # MODEL TRAINING
        # ==============================
        model = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.35).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)

        # ==============================
        # MODEL PERFORMANCE
        # ==============================
        st.subheader("📊 Model Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        c2.metric("Precision", f"{precision * 100:.2f}%")
        c3.metric("Recall", f"{recall * 100:.2f}%")
        c4.metric("Strategy", "Recall First")

        st.info(
            "The prototype prioritizes recall to reduce missed failures, "
            "while precision helps monitor false alarms."
        )

        # ==============================
        # RISK INTELLIGENCE METRICS
        # ==============================
        test_data = pd.DataFrame(X_test, columns=features)
        test_data["actual_failure"] = y_test.values
        test_data["risk_score"] = y_prob

        k = max(1, int(0.3 * len(test_data)))
        top_k = test_data.sort_values(
            by="risk_score",
            ascending=False
        ).head(k)

        actual_failures = test_data[test_data["actual_failure"] == 1]
        captured = top_k[top_k["actual_failure"] == 1]

        recall_at_k = (
            len(captured) / len(actual_failures)
            if len(actual_failures) > 0 else 0
        )

        high_risk_test = test_data[test_data["risk_score"] >= 0.70]
        false_alarms = high_risk_test[
            high_risk_test["actual_failure"] == 0
        ]

        false_alarm_rate = (
            len(false_alarms) / len(high_risk_test)
            if len(high_risk_test) > 0 else 0
        )

        lead_time = 4 + (recall_at_k * 2)

        st.subheader("⚠️ Risk Intelligence Metrics")

        m1, m2, m3 = st.columns(3)

        m1.metric("Recall@Top-K", f"{recall_at_k:.2f}")
        m2.metric("False Alarm Rate", f"{false_alarm_rate:.2f}")
        m3.metric("Estimated Lead Time", f"{lead_time:.1f} weeks")

        # ==============================
        # RISK SCORING
        # ==============================
        data["risk_score"] = model.predict_proba(X)[:, 1]
        data["risk_level"] = data["risk_score"].apply(classify_risk)
        data["risk_status"] = data["risk_level"].apply(risk_status)

        current_stress_threshold = data["current_stress"].quantile(0.75)
        load_fluctuation_threshold = data["load_fluctuation"].quantile(0.75)

        st.session_state.current_stress_threshold = current_stress_threshold
        st.session_state.load_fluctuation_threshold = load_fluctuation_threshold

        data["root_cause"] = data.apply(
            lambda row: root_cause_from_values(
                row["load"],
                row["temperature"],
                row["voltage_deviation"],
                row["current_stress"],
                row["load_fluctuation"],
                current_stress_threshold,
                load_fluctuation_threshold
            ),
            axis=1
        )

        data["maintenance_action"] = data["risk_level"].apply(
            maintenance_action
        )

        data["transformer_type"] = transformer_type

        ranked_data = data.sort_values(
            by="risk_score",
            ascending=False
        )

        st.session_state.trained_model = model
        st.session_state.features = features

        # ==============================
        # TOP 5 MAINTENANCE QUEUE
        # ==============================
        st.subheader(
            "🚨 Priority Maintenance Queue - Top 5 High-Risk Transformers"
        )

        top_5 = ranked_data[
            [
                "transformer_id",
                "transformer_type",
                "risk_score",
                "risk_status",
                "root_cause",
                "maintenance_action"
            ]
        ].head(5)

        st.dataframe(top_5)

        # ==============================
        # DOWNLOAD REPORT
        # ==============================
        csv = ranked_data[
            [
                "transformer_id",
                "transformer_type",
                "risk_score",
                "risk_status",
                "root_cause",
                "maintenance_action"
            ]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Risk-Ranked Maintenance Report",
            data=csv,
            file_name="risk_ranked_transformers.csv",
            mime="text/csv"
        )

        # ==============================
        # FEATURE IMPORTANCE
        # ==============================
        st.subheader("📌 Feature Importance")

        fig, ax = plt.subplots()
        ax.barh(features, model.feature_importances_)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # ==============================
        # RISK LEVEL DISTRIBUTION
        # ==============================
        st.subheader("📌 Risk Level Distribution")

        fig2, ax2 = plt.subplots()
        data["risk_level"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Risk Level")
        ax2.set_ylabel("Count")
        ax2.set_title("Transformer Risk Level Count")
        st.pyplot(fig2)

        # ==============================
        # CONFUSION MATRIX
        # ==============================
        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig3, ax3 = plt.subplots()
        ax3.imshow(cm)
        ax3.set_title("Confusion Matrix")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig3)

        # ==============================
        # COLORED SAMPLE RISK MAP
        # ==============================
        st.subheader("🗺️ Sample Transformer Risk Map")

        map_data = ranked_data.head(30).copy()

        np.random.seed(42)

        map_data["lat"] = 13.0 + np.random.rand(len(map_data)) * 0.3
        map_data["lon"] = 80.1 + np.random.rand(len(map_data)) * 0.3

        color_map = {
            "High": [255, 0, 0],
            "Medium": [255, 200, 0],
            "Low": [0, 180, 0]
        }

        map_data["color"] = map_data["risk_level"].map(color_map)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=1200,
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=13.08,
            longitude=80.25,
            zoom=10
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "text": (
                        "Transformer: {transformer_id}\n"
                        "Risk: {risk_level}\n"
                        "Score: {risk_score}\n"
                        "Cause: {root_cause}"
                    )
                }
            )
        )


# ======================================================
# TAB 2: MANUAL TEST MODE
# ======================================================
with tab2:

    st.subheader("🧪 Manual Single Transformer Risk Test")

    st.write(
        "Enter one transformer’s operating values and check risk instantly."
    )

    if st.button("🔄 Generate New Transformer ID"):
        st.session_state.auto_id += 1

    auto_transformer_id = f"TR-{st.session_state.auto_id}"

    left, right = st.columns(2)

    with left:

        transformer_id = st.text_input(
            "Transformer ID",
            value=auto_transformer_id
        )

        load = st.slider("Load (%)", 0, 150, 80)
        temperature = st.slider("Temperature (°C)", 0, 100, 35)
        voltage = st.slider("Voltage (V)", 150, 300, 230)
        current = st.slider("Current (A)", 0, 120, 40)
        power = st.slider("Power (kW)", 0, 60, 10)

    thermal_stress = load * temperature
    overload = int(load > 80)
    load_ratio = load / 100
    voltage_deviation = abs(voltage - 230)
    current_stress = current / (voltage + 1)
    load_fluctuation = abs(load - 70)

    model = st.session_state.trained_model
    features = st.session_state.features

    if model is not None and features is not None:

        single_input = pd.DataFrame(
            [[
                load,
                temperature,
                voltage,
                current,
                power,
                thermal_stress,
                overload,
                load_ratio,
                voltage_deviation,
                current_stress,
                load_fluctuation
            ]],
            columns=features
        )

        risk_score = model.predict_proba(single_input)[0][1]
        mode_used = "ML Model Prediction"

    else:

        risk_score = rule_based_score(
            load,
            temperature,
            voltage,
            current,
            power
        )

        mode_used = "Rule-Based Preview (Upload dataset for ML prediction)"

    risk_level = classify_risk(risk_score)

    root_cause = root_cause_from_values(
        load,
        temperature,
        voltage_deviation,
        current_stress,
        load_fluctuation,
        st.session_state.current_stress_threshold,
        st.session_state.load_fluctuation_threshold
    )

    action = maintenance_action(risk_level)

    with right:

        st.write(f"### Transformer ID: {transformer_id}")
        st.caption(mode_used)
        st.metric("Risk Score", f"{risk_score:.2f}")

        if risk_level == "Low":
            st.success("🟢 LOW RISK")
        elif risk_level == "Medium":
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

        st.write("### Probable Root Cause")
        st.write(root_cause)

        st.write("### Recommended Action")
        st.write(action)
