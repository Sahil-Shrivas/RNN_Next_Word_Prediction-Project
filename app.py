# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Load saved files
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     model = load_model("lstm_model (1).h5")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     with open("max_len.pkl", "rb") as f:
#         max_len = pickle.load(f)
#     return model, tokenizer, max_len

# model, tokenizer, max_len = load_resources()

# # ------------------------------
# # Prediction function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

#     preds = model.predict(sequence, verbose=0)
#     predicted_index = np.argmax(preds)

#     for word, index in tokenizer.word_index.items():
#         if index == predicted_index:
#             return word
#     return ""

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.set_page_config(page_title="Next Word Prediction", layout="centered")

# st.title("🧠 Next Word Prediction (LSTM)")
# st.write("Enter a sentence and the model will predict the **next word**.")

# user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

# if st.button("Predict Next Word"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         next_word = predict_next_word(user_input)
#         st.success(f"**Predicted Next Word:** {next_word}")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.caption("LSTM-based Next Word Prediction using Streamlit")










import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Next Word Prediction", layout="centered")

import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files safely
# ------------------------------
@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "lstm_model.h5")
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    maxlen_path = os.path.join(BASE_DIR, "max_len.pkl")

    # Debug (optional)
    # st.write("Files in directory:", os.listdir(BASE_DIR))

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(maxlen_path, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


# Load resources
try:
    model, tokenizer, max_len = load_resources()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return "No prediction (unknown words)"

    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')

    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds[0])

    return tokenizer.index_word.get(predicted_index, "Word not found")


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{next_word}**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")