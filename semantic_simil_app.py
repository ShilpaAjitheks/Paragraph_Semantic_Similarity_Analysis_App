#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import string
import re

scaler = MinMaxScaler()
scaler.fit(np.reshape((0, 1), (2, 1)))

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.stem import WordNetLemmatizer
    stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Failed to load NLTK resources: {e}")
    stopwords = []
    wordnet_lemmatizer = None

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    stopwords1 = nlp.Defaults.stop_words
except OSError:
    st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None
    stopwords1 = set()
except Exception as e:
    st.error(f"Failed to load spaCy: {e}")
    nlp = None
    stopwords1 = set()


def clean_text(text):
    try:
        if isinstance(text, str):
            text = text.lower()
            text = "".join([i for i in text if i not in string.punctuation])
            text = re.sub(r'\S*\d\S*\s*', '', text).strip()
        return text
    except Exception as e:
        raise ValueError(f"Error in clean_text: {e}") from e


def nltk_tokenization(text):
    try:
        if isinstance(text, str):
            if wordnet_lemmatizer is None:
                raise RuntimeError("WordNetLemmatizer not initialized.")
            tokens = re.split(r'\W+', text)
            tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens if t.lower() not in stopwords]
            return " ".join(tokens)
        return text
    except Exception as e:
        raise ValueError(f"Error in nltk_tokenization: {e}") from e


def spacy_lemmatizer(text):
    try:
        if isinstance(text, str):
            if nlp is None:
                raise RuntimeError("spaCy model not loaded.")
            doc = nlp(text)
            sent = [token.lemma_ for token in doc if token.text not in stopwords1]
            return ' '.join(sent)
        return text
    except Exception as e:
        raise ValueError(f"Error in spacy_lemmatizer: {e}") from e


@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


st.title("Semantic Similarity Analysis App")
st.markdown("By Shilpa Ajith")

try:
    from PIL import Image
    image = Image.open("semantic_similarity.png")
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.warning("Header image 'semantic_similarity.png' not found.")
except Exception as e:
    st.warning(f"Could not load image: {e}")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load sentence transformer model: {e}")
    model = None

st.subheader("Enter your text1 here:")
user_input1 = st.text_area("Input1", placeholder="Enter the first paragraph or sentence here...")
# user_input1 = st.text_area("Input1", placeholder="e.g. Deep learning models achieve superior performance on image classification tasks by learning hierarchical feature representations.")
st.subheader("Enter your text2 here:")
user_input2 = st.text_area("Input2", placeholder="Enter the second paragraph or sentence here...")
# user_input2 = st.text_area("Input2", placeholder="e.g. Neural networks with multiple layers outperform traditional methods in visual recognition by automatically extracting abstract features.")

if st.button("Predict"):
    if user_input1 and user_input2:
        try:
            if model is None:
                st.error("Model is not loaded. Cannot compute similarity.")
            else:
                p1 = clean_text(user_input1)
                p2 = clean_text(user_input2)
                p1 = nltk_tokenization(p1)
                p2 = nltk_tokenization(p2)
                p1 = spacy_lemmatizer(p1)
                p2 = spacy_lemmatizer(p2)

                if not p1.strip() or not p2.strip():
                    st.warning("Text became empty after preprocessing. Please try different input.")
                else:
                    sentence_vec1 = model.encode([p1])[0]
                    sentence_vec2 = model.encode([p2])[0]

                    similarity = cosine_similarity([sentence_vec1], [sentence_vec2])[0][0]
                    scaled_similarity = scaler.transform(np.reshape(similarity, (1, 1)))[0][0]

                    # st.header("Prediction:")
                    # st.subheader(f"{similarity:.4f}")
                    st.metric(label="Cosine Similarity Score", value=f"{similarity:.4f}")
                    # st.progress(float(similarity))

        except ValueError as e:
            st.error(f"Preprocessing error: {e}")
        except RuntimeError as e:
            st.error(f"Model or resource error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Both text fields must be filled before predicting.")
else:
    st.info("Enter both sentences above and click **Predict** to get the similarity score.")
