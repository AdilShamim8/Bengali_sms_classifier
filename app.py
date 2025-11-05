import streamlit as st
import pickle
import nltk
import string
import os
import warnings
from nltk.corpus import stopwords

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize  

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download stopwords quietly
nltk.download('stopwords', quiet=True)

# Bengali punctuation marks
bengali_punctuation = "[{(%।॥,@&*?!-…''""<#>:;)}]\n"

def bengali_stem(word):
    """Remove common Bengali suffixes from words"""
    suffixes = ["গুলো", "গুলি", "দের", "তে", "কে", "রা", "টি", "ে"]
    for suf in suffixes:
        if word.endswith(suf):
            return word[:-len(suf)]
    return word

def processed_text(text):
    """Tokenize and clean Bengali text"""
    if not text or not text.strip():
        return ""
    
    tokens = list(indic_tokenize.trivial_tokenize(text, 'bn'))
    bengali_stopwords = set(stopwords.words('bengali'))
    
    cleaned_tokens = [
        bengali_stem(token) for token in tokens
        if token not in bengali_punctuation and token not in bengali_stopwords
    ]
    return ' '.join(cleaned_tokens) if cleaned_tokens else text

def processed_sentence(model, tfidf, input_sms):
    """Process and classify SMS message"""
    proc_sent = processed_text(input_sms)
    if proc_sent.strip():
        vector_input = tfidf.transform([proc_sent])
        prob = model.predict_proba(vector_input)[0]
        confidence = prob.max()
        prediction = prob.argmax()
        return prediction, confidence
    return 0, 0.0

# Load models with error handling
@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
        model = pickle.load(open('model1.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'vectorizer1.pkl' and 'model1.pkl' are in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.stop()

tfidf, model = load_models()

# Configure page
st.set_page_config(
    page_title="বার্তাবন্ধু - Bengali SMS Classifier",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state for statistics
if 'total_classifications' not in st.session_state:
    st.session_state.total_classifications = 0
if 'class_counts' not in st.session_state:
    st.session_state.class_counts = {'normal': 0, 'promo': 0, 'spam': 0}

# Custom CSS for aesthetic design
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: 600;
        margin-bottom: 0.2em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
        font-weight: 300;
    }
    .result-box {
        padding: 1.5em;
        border-radius: 10px;
        margin: 1em 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .result-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .normal-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .promo-result {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .spam-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
st.markdown('<h1 class="main-title">বার্তাবন্ধু</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Bengali SMS Classifier</p>', unsafe_allow_html=True)

# Input section
st.markdown("### আপনার বার্তা লিখুন")
input_sms = st.text_area(
    'এসএমএস বার্তা',
    placeholder='এখানে আপনার বাংলা এসএমএস লিখুন...',
    height=150,
    label_visibility='collapsed'
)

# Add example messages with quick copy functionality
with st.expander("উদাহরণ বার্তা দেখুন"):
    col1, col2, col3 = st.columns(3)
    
    example_normal = "আসসালামু আলাইকুম। আপনি কেমন আছেন? আজ অফিসে আসবেন?"
    example_promo = "বিশেষ ছাড়! আজই কিনুন এবং ৫০% ছাড় পান! অফারটি সীমিত সময়ের জন্য।"
    example_spam = "অভিনন্দন! আপনি ১ কোটি টাকা জিতেছেন! এখনই কল করুন 01711-XXXXXX নম্বরে।"
    
    with col1:
        st.markdown("**নরমাল মেসেজ**")
        st.code(example_normal, language=None)
    with col2:
        st.markdown("**বিজ্ঞাপনমূলক**")
        st.code(example_promo, language=None)
    with col3:
        st.markdown("**স্প্যাম মেসেজ**")
        st.code(example_spam, language=None)

# Classify button
st.markdown("")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_button = st.button('শনাক্তকরণ করুন', use_container_width=True, type="primary")

if classify_button:
    if not input_sms or not input_sms.strip():
        st.warning('অনুগ্রহ করে একটি বার্তা লিখুন।')
    else:
        with st.spinner('বিশ্লেষণ করা হচ্ছে...'):
            import time
            time.sleep(0.3)  # Small delay for better UX
            result, confidence = processed_sentence(model, tfidf, input_sms)
        
        # Update statistics
        st.session_state.total_classifications += 1
        if result == 0:
            st.session_state.class_counts['normal'] += 1
        elif result == 1:
            st.session_state.class_counts['promo'] += 1
        else:
            st.session_state.class_counts['spam'] += 1
        
        st.markdown("")
        st.markdown("### শনাক্তকরণ ফলাফল")
        
        # Display result with elegant styling
        if result == 2:
            st.markdown("""
                <div class="result-box spam-result">
                    <h2 style="color: #d32f2f; margin: 0;">স্প্যাম মেসেজ</h2>
                    <p style="color: #666; margin-top: 0.5em;">এই বার্তাটি স্প্যাম হিসেবে চিহ্নিত হয়েছে। সাবধান থাকুন।</p>
                </div>
            """, unsafe_allow_html=True)
        elif result == 1:
            st.markdown("""
                <div class="result-box promo-result">
                    <h2 style="color: #f57c00; margin: 0;">বিজ্ঞাপনমূলক মেসেজ</h2>
                    <p style="color: #666; margin-top: 0.5em;">এই বার্তাটি একটি প্রচারণামূলক বার্তা।</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box normal-result">
                    <h2 style="color: #388e3c; margin: 0;">নরমাল মেসেজ</h2>
                    <p style="color: #666; margin-top: 0.5em;">এই বার্তাটি একটি স্বাভাবিক বার্তা।</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Display confidence elegantly
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            # Determine confidence color
            if confidence >= 0.9:
                conf_color = "#4caf50"  # Green
                conf_text = "অত্যন্ত নিশ্চিত"
            elif confidence >= 0.75:
                conf_color = "#2196f3"  # Blue
                conf_text = "নিশ্চিত"
            elif confidence >= 0.6:
                conf_color = "#ff9800"  # Orange
                conf_text = "মোটামুটি নিশ্চিত"
            else:
                conf_color = "#f44336"  # Red
                conf_text = "কম নিশ্চিত"
            
            st.markdown(f"""
                <div style="text-align: center; padding: 1em; background-color: #f8f9fa; border-radius: 8px; margin-top: 1em; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                    <p style="color: #666; margin: 0; font-size: 0.9em;">নিশ্চিততার মাত্রা</p>
                    <h1 style="color: {conf_color}; margin: 0.2em 0;">{confidence*100:.1f}%</h1>
                    <p style="color: #888; margin: 0; font-size: 0.85em;">{conf_text}</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(confidence)

# Footer with additional info
st.markdown("")
st.markdown("")

# Add info section in sidebar
with st.sidebar:
    st.markdown("### সিস্টেম সম্পর্কে")
    st.markdown("""
    এই সিস্টেমটি মেশিন লার্নিং ব্যবহার করে বাংলা এসএমএস বার্তা শ্রেণীবদ্ধ করে:
    
    - **নরমাল**: সাধারণ ব্যক্তিগত বার্তা
    - **বিজ্ঞাপনমূলক**: প্রচারণা/বিজ্ঞাপন বার্তা
    - **স্প্যাম**: অবাঞ্ছিত/প্রতারণামূলক বার্তা
    """)
    
    st.markdown("### প্রযুক্তি")
    st.markdown("""
    - Natural Language Processing
    - TF-IDF Vectorization
    - Ensemble Machine Learning
    - Bengali Text Processing
    """)
    
    st.markdown("### মডেল পারফরম্যান্স")
    st.metric("Accuracy", "~95%")
    st.metric("Precision", "~94%")
    
    # Session statistics
    if st.session_state.total_classifications > 0:
        st.markdown("### 📈 সেশন পরিসংখ্যান")
        st.metric("মোট শনাক্তকরণ", st.session_state.total_classifications)
        
        with st.expander("বিস্তারিত"):
            st.write(f"✅ নরমাল: {st.session_state.class_counts['normal']}")
            st.write(f"📢 বিজ্ঞাপন: {st.session_state.class_counts['promo']}")
            st.write(f"🚫 স্প্যাম: {st.session_state.class_counts['spam']}")

st.markdown(
   '<div style="text-align: center; color: #999; font-size: 0.9em; margin-top: 3em; padding: 1em; border-top: 1px solid #eee;"> Bengali SMS Classification System | Developed by  <a href="https://github.com/anikasadiaOPT" target="_blank" style="color: #999;">Sadia Afrin Anika</a> • Contributions by  <a href="https://github.com/AdilShamim8" target="_blank" style="color: #999;">Adil Shamim</a> </div>',
unsafe_allow_html=True
)
