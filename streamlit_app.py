import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION & STYLING
# ============================================
st.set_page_config(
    page_title="🐾 Animal Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful design
st.markdown("""
    <style>
    /* Main background and text */
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide default header and footer */
    header {
        visibility: hidden;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title styling */
    h1 {
        text-align: center;
        color: #2a2d5a !important;
        font-size: 3.5em !important;
        font-weight: 900 !important;
        margin-bottom: 0.5em !important;
        letter-spacing: 2px !important;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.8em;
        color: #555;
        margin-bottom: 2em;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
        animation: slideIn 0.6s ease-out;
    }
    
    .animal-name {
        font-size: 3em;
        font-weight: 900;
        margin: 20px 0;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .confidence-score {
        font-size: 2.5em;
        font-weight: 700;
        margin: 15px 0;
    }
    
    .confidence-label {
        font-size: 1em;
        opacity: 0.9;
    }
    
    /* Upload area styling */
    .upload-container {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    /* Top predictions styling */
    .top-predictions {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .prediction-item {
        background: linear-gradient(90deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        font-weight: 600;
    }
    
    /* Image preview styling */
    .image-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3d4b7d 0%, #4a3a7a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    </style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR INFORMATION
# ============================================
with st.sidebar:
    st.markdown("# 📋 Information")
    st.markdown("---")
    
    st.markdown("## 🦁 Supported Animals")
    animals_emoji = {
        "🐕 Dog": "dog",
        "🐴 Horse": "horse",
        "🐘 Elephant": "elephant",
        "🦋 Butterfly": "butterfly",
        "🐔 Chicken": "chicken",
        "🐱 Cat": "cat",
        "🐄 Cow": "cow",
        "🐑 Sheep": "sheep",
        "🕷️ Spider": "spider",
        "🐿️ Squirrel": "squirrel"
    }
    
    for emoji_name, _ in animals_emoji.items():
        st.markdown(f"• {emoji_name}")
    
    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.markdown("""
    - Use **clear, well-lit** images
    - Ensure the animal is **clearly visible**
    - Use **similar** images to training data
    - **Center** the animal in the frame
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 20px;">
        <small>🚀 Built with Streamlit & TensorFlow</small><br>
        <small>Animals-10 AI Classifier</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("model 9.keras")

try:
    model = load_my_model()
except Exception as e:
    st.error("❌ Could not find 'model 9.keras'. Please make sure the model file is in the same directory.")
    st.stop()

# ============================================
# ANIMAL CLASSES
# ============================================
ENGLISH_CLASSES = [
    'dog',        # cane (0)
    'horse',      # cavallo (1)
    'elephant',   # elefante (2)
    'butterfly',  # farfalla (3)
    'chicken',    # gallina (4)
    'cat',        # gatto (5)
    'cow',        # mucca (6)
    'sheep',      # pecora (7)
    'spider',     # ragno (8)
    'squirrel'    # scoiattolo (9)
]

# ============================================
# MAIN HEADER
# ============================================
st.markdown("""
    <div>
        <h1>🐾 Animal Classifier</h1>
        <p class="subtitle">✨ Upload an image to identify the animal using AI ✨</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# FILE UPLOADER
# ============================================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📤 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to select an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an animal"
    )

with col2:
    st.markdown("### 👁️ Preview")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)
    else:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 60px 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
            ">
                <h3 style="margin: 0;">No Image Yet</h3>
                <p style="margin: 10px 0 0 0;">Upload to see preview</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================
# PREDICTION LOGIC
# ============================================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    with st.spinner("🔍 Analyzing image..."):
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized.convert("RGB"))
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Inference
        predictions = model.predict(img_tensor, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[::-1][:3]
        top_3_predictions = [(ENGLISH_CLASSES[i], predictions[0][i]) for i in top_3_indices]
    
    st.markdown("---")
    
    # ============================================
    # MAIN PREDICTION RESULT
    # ============================================
    st.markdown(f"""
        <div class="prediction-box">
            <div class="animal-name">{ENGLISH_CLASSES[predicted_idx]}</div>
            <div class="confidence-score">{confidence * 100:.1f}%</div>
            <div class="confidence-label">Confidence Score</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # TOP 3 PREDICTIONS
    # ============================================
    st.markdown("### 🏆 Top 3 Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (animal, score) in enumerate(top_3_predictions):
        with [col1, col2, col3][idx]:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                ">
                    <h3 style="margin: 0; font-size: 1.2em;">#{idx + 1}</h3>
                    <h2 style="margin: 10px 0 0 0; font-size: 1.5em;">{animal.upper()}</h2>
                    <h3 style="margin: 10px 0 0 0; font-size: 2em; font-weight: 900;">{score * 100:.1f}%</h3>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # CONFIDENCE DISTRIBUTION CHART
    # ============================================
    st.markdown("### 📊 All Predictions")
    
    # Create plotly chart
    fig = go.Figure(data=[
        go.Bar(
            x=[ENGLISH_CLASSES[i] for i in range(len(ENGLISH_CLASSES))],
            y=predictions[0] * 100,
            marker=dict(
                color=predictions[0],
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"{score * 100:.1f}%" for score in predictions[0]],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores for All Animals",
        xaxis_title="Animal",
        yaxis_title="Confidence (%)",
        height=400,
        template="plotly_white",
        font=dict(size=12),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================
    # ACTION BUTTONS
    # ============================================
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Try Another Image", use_container_width=True, key="retry"):
            st.rerun()

else:
    # ============================================
    # WELCOME MESSAGE
    # ============================================
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            margin-top: 30px;
        ">
            <h2 style="margin: 0; font-size: 2em;">👆 Get Started</h2>
            <p style="margin: 15px 0 0 0; font-size: 1.1em;">Upload an animal image to see the AI magic!</p>
        </div>
    """, unsafe_allow_html=True)
