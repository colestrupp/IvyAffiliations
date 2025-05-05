import streamlit as st
import joblib
import random
import numpy as np

# Load model & vectorizer
model = joblib.load("affiliation_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Ivy Affiliation Akinator 🎓")
st.subheader("Enter a name to predict their affiliation.")
st.image("https://theivyclub.org/wp-content/uploads/2023/07/LogoIvyTrGold.png", width=400)

with st.form(key="input_form"):
    name_input = st.text_input("Enter a name:")
    submit_button = st.form_submit_button("Predict")

if submit_button:
    if name_input:
        name_vectorized = vectorizer.transform([name_input])
        probabilities = model.predict_proba(name_vectorized)

        if max(probabilities[0]) < 0.3:  # If confidence is low

            valid_indices = np.where(probabilities[0] > 0.05)[0]
            options = [model.classes_[i] for i in valid_indices]
            prediction = random.choice(options)  # Randomly choose another label
        else:
            prediction = model.predict(vectorizer.transform([name_input]))[0]

        st.session_state.latest_entry = f"**{name_input} →** {prediction}"

# Display the latest prediction (if available)
if "latest_entry" in st.session_state:
    st.write(st.session_state.latest_entry)
    st.write("*NOTE: for research purposes only*")
