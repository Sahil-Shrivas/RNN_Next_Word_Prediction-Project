
# import streamlit as st

# # ✅ MUST BE FIRST STREAMLIT COMMAND
# st.set_page_config(page_title="Next Word Prediction", layout="centered")

# import pickle
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Load saved files safely
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#     model_path = os.path.join(BASE_DIR, "lstm_model.h5")
#     tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
#     maxlen_path = os.path.join(BASE_DIR, "max_len.pkl")

#     # Debug (optional)
#     # st.write("Files in directory:", os.listdir(BASE_DIR))

#     model = load_model(model_path)

#     with open(tokenizer_path, "rb") as f:
#         tokenizer = pickle.load(f)

#     with open(maxlen_path, "rb") as f:
#         max_len = pickle.load(f)

#     return model, tokenizer, max_len


# # Load resources
# try:
#     model, tokenizer, max_len = load_resources()
# except Exception as e:
#     st.error(f"Error loading files: {e}")
#     st.stop()

# # ------------------------------
# # Prediction function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]

#     if len(sequence) == 0:
#         return "No prediction (unknown words)"

#     sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')

#     preds = model.predict(sequence, verbose=0)
#     predicted_index = np.argmax(preds[0])

#     return tokenizer.index_word.get(predicted_index, "Word not found")


# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.title("🧠 Next Word Prediction (LSTM)")
# st.write("Enter a sentence and the model will predict the **next word**.")

# user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

# if st.button("Predict Next Word"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         next_word = predict_next_word(user_input)
#         st.success(f"Predicted Next Word: **{next_word}**")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.caption("LSTM-based Next Word Prediction using Streamlit")

































import streamlit as st

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Next Word Prediction",
    layout="centered"
)

import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load Resources
# ------------------------------
@st.cache_resource
def load_resources():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "lstm_model.h5")
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    maxlen_path = os.path.join(BASE_DIR, "max_len.pkl")

    # Check files exist
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()

    if not os.path.exists(tokenizer_path):
        st.error("Tokenizer file not found!")
        st.stop()

    if not os.path.exists(maxlen_path):
        st.error("max_len.pkl file not found!")
        st.stop()

    # Load model
    model = load_model(model_path, compile=False)

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load max length
    with open(maxlen_path, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


# ------------------------------
# Load Everything
# ------------------------------
try:
    model, tokenizer, max_len = load_resources()

except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()


# ------------------------------
# Prediction Function
# ------------------------------
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return "No prediction"

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding='pre'
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_word = ""

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            predicted_word = word
            break

    return predicted_word


# ------------------------------
# UI
# ------------------------------
st.title("🧠 Next Word Prediction App")

st.write(
    "Enter a sentence and the LSTM model will predict the next word."
)

user_input = st.text_input(
    "✍️ Enter your sentence:"
)

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        next_word = predict_next_word(user_input)

        st.success(f"Predicted Next Word: {next_word}")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.caption(
    "Built with ❤️ using TensorFlow, LSTM & Streamlit"
)

