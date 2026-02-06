import streamlit as st
import pickle
import numpy as np
import os
import re
from collections import Counter
import nltk

# Download NLTK data quietly (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ---------- MINIMAL WORD2VEC (PURE NUMPY - NO GENSIM) ----------
class Word2VecNumPy:
    """Minimal inference-only version for Streamlit deployment"""
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.word2idx = {}
        self.vectors = None
    
    def __getitem__(self, word):
        """Enable model[word] syntax"""
        if word in self.word2idx and self.vectors is not None:
            return self.vectors[self.word2idx[word]]
        return np.zeros(self.vector_size)
    
    def __contains__(self, word):
        return word in self.word2idx
    
    @classmethod
    def load(cls, path):
        """Load pre-trained model from pickle file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Word2Vec model not found at: {path}\n"
                                  f"Please train the model first using the training script.")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(vector_size=data['vector_size'])
        model.word2idx = data['word2idx']
        model.vectors = data['vectors']
        return model

# ---------- TEXT PREPROCESSING (NLTK-BASED) ----------
def clean_text(text):
    """Clean and tokenize text using NLTK (Windows-safe)"""
    # Clean: lowercase + keep only letters/spaces
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Tokenize with NLTK
    if cleaned:
        return nltk.word_tokenize(cleaned)
    return []

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Flipkart Sentiment Analysis", page_icon="🛒")
st.title("🛒 Flipkart Sentiment Analysis (Classic NLP)")

# Feature selection with visual icons
feature_type = st.selectbox(
    "🔤 Choose Feature Extraction Method",
    ["Bag-of-Words (BoW)", "TF-IDF", "Word2Vec (Pure NumPy)"],
    format_func=lambda x: x
)

review = st.text_area(
    "✍️ Enter Product Review",
    placeholder="e.g., 'This tawa is excellent quality and non-stick!'",
    height=100
)

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("⚠️ Please enter a review first!")
        st.stop()
    
    # Preprocess text
    tokens = clean_text(review)
    text = " ".join(tokens)
    
    if not tokens:
        st.error("❌ No valid words found in review after cleaning!")
        st.stop()
    
    # Show preprocessing results
    with st.expander("🔍 Preprocessing Details", expanded=False):
        st.write(f"**Original:** {review}")
        st.write(f"**Cleaned tokens:** {tokens}")
        st.write(f"**Vocabulary size:** {len(tokens)} words")
    
    try:
        if feature_type == "Bag-of-Words (BoW)":
            # Load BoW vectorizer and classifier
            vectorizer = pickle.load(open("features/bow_features.pkl", "rb"))
            model = pickle.load(open("models/bow_model.pkl", "rb"))
            X = vectorizer.transform([text])
            method = "BoW"
            
        elif feature_type == "TF-IDF":
            # Load TF-IDF vectorizer and classifier
            vectorizer = pickle.load(open("features/tfidf_features.pkl", "rb"))
            model = pickle.load(open("models/tfidf_model.pkl", "rb"))
            X = vectorizer.transform([text])
            method = "TF-IDF"
            
        else:  # Word2Vec (Pure NumPy)
            # Load classifier (scikit-learn model)
            model = pickle.load(open("models/w2v_model.pkl", "rb"))
            # Load PURE NUMPY Word2Vec embeddings (NO GENSIM!)
            w2v_model = Word2VecNumPy.load("features/w2v_model.model")
            
            # Generate document vector
            vecs = [w2v_model[word] for word in tokens if word in w2v_model]
            if vecs:
                vec = np.mean(vecs, axis=0)
            else:
                vec = np.zeros(100)  # Fallback for OOV words
            
            X = vec.reshape(1, -1)
            method = "Word2Vec (NumPy)"
        
        # Make prediction
        prediction = model.predict(X)[0]
        proba = (model.predict_proba(X)[0] * 100).max() if hasattr(model, "predict_proba") else None
        
        # Display results with visual feedback
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.success(f"✅ **POSITIVE** ({proba:.1f}%)" if proba else "✅ **POSITIVE**")
                st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/grinning-face-with-smiling-eyes_1f604.png", width=100)
            else:
                st.error(f"❌ **NEGATIVE** ({proba:.1f}%)" if proba else "❌ **NEGATIVE**")
                st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/325/pensive-face_1f614.png", width=100)
        
        with col2:
            st.markdown(f"### 📊 Analysis Details")
            st.write(f"**Method:** {method}")
            st.write(f"**Tokens processed:** {len(tokens)}")
            st.write(f"**Known words:** {len([w for w in tokens if w in w2v_model]) if 'w2v_model' in locals() else 'N/A'}")
            
            # Show word contributions for Word2Vec
            if feature_type == "Word2Vec (Pure NumPy)" and vecs:
                with st.expander("💡 Word Contributions (Top 5)", expanded=False):
                    word_scores = [(w, np.linalg.norm(w2v_model[w])) for w in tokens if w in w2v_model]
                    word_scores.sort(key=lambda x: x[1], reverse=True)
                    for word, score in word_scores[:5]:
                        st.write(f"`{word}` → strength: {score:.3f}")
    
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found:\n`{str(e)}`")
        st.info("""
        **To fix this:**
        1. Run the training script first to generate model files
        2. Ensure these directories exist:
           - `features/` (for vectorizers/embeddings)
           - `models/` (for classifiers)
        """)
        st.stop()
    
    except Exception as e:
        st.exception(f"Unexpected error: {str(e)}")
        st.stop()

# Footer with usage tips
st.markdown("---")
st.caption("💡 **Tip:** Try reviews like 'excellent product' (positive) or 'terrible quality' (negative) to see the model in action!")
st.caption("🔒 **Note:** This app uses pure NumPy Word2Vec - no Gensim/PyTorch dependencies = no Windows installation issues!")

# Sidebar with model info
with st.sidebar:
    st.header("⚙️ Model Info")
    st.write("**Feature Methods:**")
    st.write("- BoW: Simple word counting")
    st.write("- TF-IDF: Weighted word importance")
    st.write("- Word2Vec: Semantic word embeddings")
    st.write("\n**Training Data:**")
    st.write("Flipkart product reviews")
    st.write("(tawas, shuttlecocks, electronics)")
    st.write("\n**Pure NumPy Implementation**")
    st.write("✅ No Gensim")
    st.write("✅ No binary conflicts")
    st.write("✅ Windows-friendly")