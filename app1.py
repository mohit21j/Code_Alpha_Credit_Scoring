import streamlit as st
import numpy as np
import pickle

# Load files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("categorical_mappings.pkl", "rb") as f:
    categorical_mappings = pickle.load(f)

# Create reverse mappings
category_to_code = {
    col: {v: i for i, v in enumerate(options)}
    for col, options in categorical_mappings.items()
}

# Streamlit layout
st.set_page_config(page_title="Credit Score Predictor", layout="centered")
st.title("💳 Credit Score Prediction App")

user_input = []
for feature in feature_names:
    if feature in categorical_mappings:
        selected = st.selectbox(f"{feature}", categorical_mappings[feature])
        encoded = category_to_code[feature][selected]
        user_input.append(encoded)
    else:
        value = st.number_input(f"{feature}", step=1.0)
        user_input.append(value)

# Predict
if st.button("Predict Credit Score"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    if isinstance(prediction, (int, float)):
        if prediction >= 700:
            label = "Good"
            color = "green"
        elif prediction >= 500:
            label = "Average"
            color = "orange"
        else:
            label = "Poor"
            color = "red"
    else:
        label = prediction
        color = "blue"

    st.markdown(f"### 🧠 Credit Score Level: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)