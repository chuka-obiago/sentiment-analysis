import streamlit as st
import joblib
import os

# Load the trained SVM model (make sure it's in the same directory or adjust path)
@st.cache_resource
def load_model():
    model_path = "svm_sentiment_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload svm_sentiment_model.pkl")
        return None
    return joblib.load(model_path)

model = load_model()

# Set up the page
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title('💬 Sentiment Analysis App')
st.write("This app predicts whether your feedback is **Positive**, **Neutral**, or **Negative** using machine learning.")

# Input field
user_input = st.text_area("Type your feedback or review here:", height=150, placeholder="e.g. The rice is very nice")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    elif model:
        prediction = model.predict([user_input])[0]
        
        if prediction == "positive":
            st.success(f"✅ Sentiment: **Positive**")
        elif prediction == "neutral":
            st.info(f"😐 Sentiment: **Neutral**")
        else:
            st.error(f"❌ Sentiment: **Negative**")
