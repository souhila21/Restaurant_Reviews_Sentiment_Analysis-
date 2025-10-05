# app.py
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from dotenv import load_dotenv
import os
# The correct library for your structure is google.generativeai
import google.generativeai as genai
import json # Used for parsing the guaranteed JSON output

# load API Key
load_dotenv()
genai.configure(api_key = os.getenv('GEMINI_API_KEY'))

# ✅ Put this here — first Streamlit command
st.set_page_config(page_title="Restaurant Review Classifier", layout="centered")

# Ensure NLTK data is available (downloads only if missing)
try:
    nltk.download("stopwords", quiet=True)
except Exception:
    st.warning("Could not download NLTK stopwords. Check network connection.")


@st.cache_resource
def load_model_and_vectorizer(model_path="Restaurant_review_model.pkl", vectorizer_path="count_v_res.pkl"):
    """
    Loads the cached ML model and vectorizer.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def preprocess_text(text: str) -> str:
    custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                        'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english")) - custom_stopwords

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    words = review.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)


def gemini_predict(text, ml_confidence):
    """
    Calls the Gemini API using structured output (JSON mode) for reliable parsing.
    The response configuration is passed to the GenerativeModel constructor to fix the TypeError.
    """

    # 1. Define the Structured Output Schema (JSON Schema format)
    response_schema = {
        "type": "object",
        "properties": {
            "Sentiment": {"type": "string", "description": "Classify as Positive, Negative, or Neutral."},
            "Advice": {"type": "string", "description": "The short, actionable recommendation (1-2 sentences)."},
        },
        "required": ["Sentiment", "Advice"],
    }

    # 2. Define the Configuration (passed to constructor in this SDK version)
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        # Other optional configs like temperature can go here: "temperature": 0.2
    }

    # 3. Initialize the Generative Model with the config (FIX for TypeError)
    # Using gemini-2.5-flash for speed and cost-efficiency
    model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)

    # 4. Construct the Prompt with context (including ML confidence)
    if ml_confidence >= 0.95:
        ml_hint = "The machine learning model is highly confident (P > 0.95) that this is a positive review."
    elif ml_confidence <= 0.05:
        ml_hint = "The machine learning model is highly confident (P < 0.05) that this is a negative review."
    else:
        ml_hint = "The machine learning model is uncertain (P is near 0.50). You may classify it as Neutral if the content is ambiguous."

    prompt = f"""
    You are a helpful restaurant assistant specializing in customer response. 
    Your goal is to analyze the review and provide actionable advice.

    ML Model Hint: {ml_hint}

    Analyze the following restaurant review:
    {text}

    1️⃣ Classify the sentiment as **Positive**, **Negative**, or **Neutral**. 
    2️⃣ Give a short, friendly recommendation (1–2 sentences) on how the restaurant should respond to ensure customer satisfaction.
    (If positive: suggest thanking and encouraging a return. If negative: suggest an apology and specific fix/offer. If neutral: suggest engaging the customer for more details.)

    Return the final JSON object only.
    """

    # 5. Call generate_content (without the 'config' keyword argument)
    response = model.generate_content(prompt)

    # 6. Parse the guaranteed JSON output
    try:
        # The model is forced to output JSON, so we can reliably parse response.text
        data = json.loads(response.text)
        sentiment = data.get("Sentiment", "N/A")
        advice = data.get("Advice", "Could not generate advice.")
        return sentiment, advice
    except json.JSONDecodeError:
        # Fallback if the model somehow failed to output valid JSON
        st.error(f"Error: Could not parse Gemini's JSON response: {response.text}")
        return "N/A", "Please try again. Gemini's output was corrupt."


def sentiment_badge(sentiment: str) -> str:
    """Return HTML badge with color-coded sentiment."""
    sentiment_lower = sentiment.strip().lower()
    color = "#2ecc71"  # green default

    if sentiment_lower == "negative":
        color = "#e74c3c"  # red
    elif sentiment_lower == "neutral":
        color = "#f39c12"  # orange

    return f"<span style='background-color:{color}; color:white; padding:6px 12px; border-radius:8px;'>{sentiment.title()}</span>"


def main():
    st.title("🍽️ Restaurant Review Classifier")
    st.subheader("Reviews Analysis powered by Gemini AI")

    st.write("Enter a restaurant review below and click **Classify** to predict sentiment.")

    review_input = st.text_area("Your review", height=160, placeholder="Type the review here...")

    # Sidebar for file inputs
    model_path = st.sidebar.text_input("ML Model file path", value="Restaurant_review_model.pkl")
    vectorizer_path = st.sidebar.text_input("Vectorizer file path", value="count_v_res.pkl")
    st.sidebar.caption("Put your saved model and CountVectorizer filenames here (joblib).")

    # Load model & vectorizer once and cache
    try:
        model, vectorizer = load_model_and_vectorizer(model_path=model_path, vectorizer_path=vectorizer_path)
    except Exception as e:
        st.sidebar.error(f"Error loading model/vectorizer: {e}")
        st.stop()

    if st.button("Classify"):
        if not review_input or review_input.strip() == "":
            st.warning("Please enter a review before clicking Classify.")
        else:
            try:
                # 1. ML Preprocessing & Vectorization
                processed = preprocess_text(review_input)
                X_vec = vectorizer.transform([processed])

                # 2. ML Prediction & Confidence
                # Get probability for both classes: [P(class 0, negative), P(class 1, positive)]
                proba = model.predict_proba(X_vec)[0]
                pred = model.predict(X_vec)[0]

                sentiment_ml = "Positive" if pred == 1 else "Negative"

                # Confidence is the maximum probability
                confidence = max(proba)

                # Pass P(Positive) to Gemini for contextual prompting
                ml_confidence_for_gemini = proba[1]

                # 3. Gemini Analysis
                sentiment_gemini, advice_gemini = gemini_predict(review_input, ml_confidence_for_gemini)

                # --- UI OUTPUT ---

                st.markdown("---")

                # Use columns for a cleaner layout
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("🤖 ML Model Prediction")
                    st.markdown(sentiment_badge(sentiment_ml), unsafe_allow_html=True)
                    st.write(f"Confidence: **{confidence:.2%}**")

                with col2:
                    st.subheader("🔍 Gemini AI Sentiment")
                    st.markdown(sentiment_badge(sentiment_gemini), unsafe_allow_html=True)
                    # Show warning if the ML model result differs from Gemini (e.g., Negative vs Neutral)
                    if sentiment_ml.lower() != sentiment_gemini.lower():
                        st.warning("Discrepancy: ML model (Binary) differs from Gemini AI (can be Neutral).")

                st.markdown("---")
                st.subheader("💡 Actionable Advice")
                st.info(f"**Recommendation:** {advice_gemini}")

                # 4. Transparency
                with st.expander("Show ML Model Details"):
                    st.write("**Processed text sent to ML model:**")
                    st.code(processed)

            except Exception as e:
                # Catch general ML or vectorizer errors
                st.error(f"Error during core analysis: {e}")
                # st.exception(e) # Keep this commented unless you need full traceback for debugging


if __name__ == "__main__":
    main()