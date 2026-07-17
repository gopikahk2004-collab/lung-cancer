import os
# Force CPU execution to avoid TensorFlow Metal device prediction bugs on macOS
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="Lung Cancer Histology Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom medical dashboard CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply Outfit font globally */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header customization */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Card design */
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    /* Classification labels */
    .diagnosis-normal {
        background-color: #DCFCE7;
        color: #166534;
        border-left: 5px solid #22C55E;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .diagnosis-aca {
        background-color: #FEF3C7;
        color: #92400E;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .diagnosis-scc {
        background-color: #FEE2E2;
        color: #991B1B;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Initialization & Caching Model
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lung_cancer_cnn.keras")

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None

model = get_model()

# Configuration and mapping
CLASSES = {
    0: {"name": "Lung Adenocarcinoma (ACA)", "key": "lung_aca", "type": "Malignant"},
    1: {"name": "Normal Lung Tissue", "key": "lung_n", "type": "Benign"},
    2: {"name": "Lung Squamous Cell Carcinoma (SCC)", "key": "lung_scc", "type": "Malignant"}
}

# Image parameters
IMG_SIZE = (128, 128)

# ----------------------------------------------------
# 3. Sidebar Configuration
# ----------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/lungs.png", width=80)
st.sidebar.markdown("<h2 style='margin-top:0;'>Histology Lab Portal</h2>", unsafe_allow_html=True)
st.sidebar.write("Using Convolutional Neural Network (CNN) to detect pathology from histology slide cuts.")

# Session State for navigation page redirection
if "page" not in st.session_state:
    st.session_state.page = "🔬 Diagnosis Lab"

# Session State for uploader redirection or sample inputs
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

pages = ["🔬 Diagnosis Lab", "📚 Histology Education", "🖼️ Local Sample Gallery"]
try:
    default_idx = pages.index(st.session_state.page)
except ValueError:
    default_idx = 0

selected_page = st.sidebar.radio("Navigation", pages, index=default_idx)
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

st.sidebar.divider()

# Model Metadata in sidebar
st.sidebar.subheader("CNN Model Details")
with st.sidebar.container(border=True):
    st.markdown("""
    * **Architecture**: Sequential CNN
    * **Input Dimensions**: 128x128x3 (RGB)
    * **Output Layer**: Dense (3 units, Softmax)
    * **Classification Classes**: 3 classes
    * **Device Engine**: CPU Safe-Mode
    """)

# Helper function to predict
def predict_image(image_input):
    if model is None:
        return None
    # Preprocess image
    img = image_input.convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0)  # Shape (1, 128, 128, 3)
    
    # Run prediction
    preds = model.predict(img_batch, verbose=0)[0]
    return preds

# ----------------------------------------------------
# 4. Main Page Routes
# ----------------------------------------------------

# PAGE 1: DIAGNOSIS LAB
if st.session_state.page == "🔬 Diagnosis Lab":
    st.markdown("<h1 class='main-title'>🔬 Lung Cancer Diagnosis Lab</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload histological images or load local database slices for CNN classification.</p>", unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 1.1], gap="large")
    
    with col_input:
        st.subheader("Image Input Selection")
        
        # Choice of input: Upload or Sample select
        input_type = st.radio(
            "Choose source format:",
            ["Upload Your Own File", "Choose from Local Samples"],
            index=1 if st.session_state.selected_sample is not None else 0
        )
        
        selected_img = None
        img_filename = ""
        
        if input_type == "Upload Your Own File":
            st.session_state.selected_sample = None  # Reset sample state
            uploaded_file = st.file_uploader("Upload histopathology JPEG/PNG image:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    selected_img = Image.open(uploaded_file)
                    img_filename = uploaded_file.name
                except Exception as e:
                    st.error(f"Invalid image file: {e}")
        else:
            # Choose from Local Samples
            st.markdown("Select a class and file from the workspace dataset directories:")
            c_col, f_col = st.columns(2)
            
            # Select class
            class_options = ["lung_aca (Adenocarcinoma)", "lung_n (Normal)", "lung_scc (Squamous Carcinoma)"]
            
            # Use preselected class if set by Gallery redirection
            preselected_class_idx = 0
            preselected_file_name = None
            
            if st.session_state.selected_sample:
                cls_key = st.session_state.selected_sample["class"]
                preselected_file_name = st.session_state.selected_sample["filename"]
                if cls_key == "lung_n":
                    preselected_class_idx = 1
                elif cls_key == "lung_scc":
                    preselected_class_idx = 2
                    
            selected_class_str = c_col.selectbox("Tissue Class Folder", class_options, index=preselected_class_idx)
            class_key = "lung_aca" if "lung_aca" in selected_class_str else ("lung_n" if "lung_n" in selected_class_str else "lung_scc")
            
            # Get list of images in that directory
            dir_path = os.path.join(BASE_DIR, class_key)
            if os.path.exists(dir_path):
                img_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            else:
                img_files = []
                st.error(f"Sample folder not found: {dir_path}")
                
            if img_files:
                # Find index of preselected file if valid
                default_file_idx = 0
                if preselected_file_name and preselected_file_name in img_files:
                    default_file_idx = img_files.index(preselected_file_name)
                
                selected_file = f_col.selectbox("Select Sample Image", img_files, index=default_file_idx)
                
                if selected_file:
                    img_path = os.path.join(dir_path, selected_file)
                    try:
                        selected_img = Image.open(img_path)
                        img_filename = f"{class_key}/{selected_file}"
                    except Exception as e:
                        st.error(f"Error loading sample image: {e}")
            else:
                st.warning("No images found in the selected folder.")
        
        # Display image preview
        if selected_img is not None:
            st.divider()
            st.markdown("**Image Preview:**")
            st.image(selected_img, width='stretch', caption=img_filename)
            
            # Predict Button
            run_btn = st.button("Run CNN Pathology Analysis", type="primary", width='stretch')
        else:
            st.info("Please select or upload a tissue slice image to trigger diagnosis.")
            run_btn = False

    with col_result:
        st.subheader("Diagnostic Report")
        
        if run_btn and selected_img is not None:
            if model is None:
                st.error("Cannot execute diagnosis. Model not loaded.")
            else:
                with st.spinner("Analyzing image features, extracting CNN dimensions..."):
                    # Process prediction
                    probabilities = predict_image(selected_img)
                    
                if probabilities is not None:
                    predicted_idx = np.argmax(probabilities)
                    pred_class_info = CLASSES[predicted_idx]
                    confidence = probabilities[predicted_idx]
                    
                    # Custom Colored Banners depending on diagnosis
                    if predicted_idx == 0:  # Adenocarcinoma
                        st.markdown(f"""
                        <div class="diagnosis-aca">
                            <span style="font-size:1.4rem;">⚠️ DETECTION: {pred_class_info['name']}</span><br>
                            Status: MALIGNANT | Severity Level: HIGH | Confidence: {confidence*100:.2f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("**Adenocarcinoma (ACA)** is a malignant epithelial neoplasm with glandular differentiation. It typically arises from the mucus-secreting cells of the lung lining. Highly recommended to seek formal oncological pathological review.")
                        
                    elif predicted_idx == 1:  # Normal Tissue
                        st.markdown(f"""
                        <div class="diagnosis-normal">
                            <span style="font-size:1.4rem;">✅ DETECTION: {pred_class_info['name']}</span><br>
                            Status: BENIGN | Pathology: NONE DETECTED | Confidence: {confidence*100:.2f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success("The tissue sample displays healthy lung cells. The slide demonstrates clear, open alveolar airspace partitions, thin pneumocyte walls, and typical capillary cells with no signs of malignant overcrowding.")
                        
                    elif predicted_idx == 2:  # Squamous Carcinoma
                        st.markdown(f"""
                        <div class="diagnosis-scc">
                            <span style="font-size:1.4rem;">🚨 DETECTION: {pred_class_info['name']}</span><br>
                            Status: MALIGNANT | Severity Level: CRITICAL | Confidence: {confidence*100:.2f}%
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.error("**Squamous Cell Carcinoma (SCC)** is a malignant epithelial neoplasm showing keratinization and/or intercellular bridges. It is strongly linked to chronic airway irritation, such as cigarette smoke. Immediate medical path correlation recommended.")
                    
                    # Bar Chart of Softmax Probabilities
                    st.write("**Classification Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Class": [CLASSES[i]["name"] for i in range(3)],
                        "Probability": [float(p) for p in probabilities],
                        "Color": ["#F59E0B", "#22C55E", "#EF4444"]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        range_x=[0, 1],
                        color="Class",
                        color_discrete_map={
                            CLASSES[0]["name"]: "#F59E0B", # Amber
                            CLASSES[1]["name"]: "#22C55E", # Green
                            CLASSES[2]["name"]: "#EF4444"  # Red
                        },
                        text="Probability"
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=10),
                        xaxis=dict(tickformat=".1%"),
                        yaxis=dict(title="")
                    )
                    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Detail breakdown cards
                    st.markdown("### Tissue Metric Analysis")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("ACA Probability", f"{probabilities[0]*100:.1f}%")
                    with c2:
                        st.metric("Normal Probability", f"{probabilities[1]*100:.1f}%")
                    with c3:
                        st.metric("SCC Probability", f"{probabilities[2]*100:.1f}%")
                        
                else:
                    st.error("Error occurred while executing predictions.")
        else:
            st.info("Diagnose results will display here once the neural network model finishes scanning the input image.")


# PAGE 2: HISTOLOGY EDUCATION
elif st.session_state.page == "📚 Histology Education":
    st.markdown("<h1 class='main-title'>📚 Lung Pathology Histology Guide</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Learn about the architectural features that separate cancerous cells from normal lung tissue.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Histopathology is the study of tissue changes caused by disease. Under the microscope, pathologists analyze the shape, structure, and arrangement of cells to diagnose conditions. 
    Our CNN model is trained to differentiate between three specific conditions:
    """)
    
    # 3 Column explanation
    col_aca, col_n, col_scc = st.columns(3, gap="medium")
    
    with col_aca:
        st.subheader("1. Adenocarcinoma (ACA)")
        st.markdown("""
        **What it is:**  
        Adenocarcinoma is a non-small cell lung cancer (NSCLC) originating in glandular cells, which produce fluids like mucus. It is the most common lung cancer in non-smokers and women.
        
        **Key Architectural Clues:**
        * **Glandular Formations:** Cells organize into circles or tubes mimicking glands.
        * **Mucin Secretion:** Abundant intracellular mucus/mucin might push nuclei to the side.
        * **Atypical Nuclei:** Nuclei are large, dark (hyperchromatic), and vary widely in size.
        """)
        
    with col_n:
        st.subheader("2. Normal Lung Tissue (N)")
        st.markdown("""
        **What it is:**  
        Healthy lung tissue optimized for gaseous exchange. It is characterized by thin, spacious structures.
        
        **Key Architectural Clues:**
        * **Alveolar Air Spaces:** Delicate, lace-like structures resembling open bubbles (alveoli).
        * **Thin Epithelial Walls:** Air sacs are lined by simple, flat cells (Type I and II pneumocytes).
        * **Low Cellular Density:** The slide is mostly comprised of open air space rather than dense cells.
        """)
        
    with col_scc:
        st.subheader("3. Squamous Carcinoma (SCC)")
        st.markdown("""
        **What it is:**  
        Another primary NSCLC type, SCC develops in the flat squamous cells lining the inner airways (bronchi). It is very strongly linked to cigarette smoke.
        
        **Key Architectural Clues:**
        * **Keratin Pearls:** Concentric layers of dense pink keratin protein produced by cancer cells.
        * **Intercellular Bridges:** Thin lines (desmosomes) visible between crowded tumor cells.
        * **High Cellular Density:** Sheets of large, irregular cells aggressively growing and displacing open spaces.
        """)
        
    st.divider()
    
    st.subheader("At-A-Glance Histological Comparison Table")
    
    comparison_data = {
        "Attribute": [
            "Anatomical Location", 
            "Cell Origin", 
            "Microscopic Structure", 
            "Air Spaces (Alveoli)", 
            "Smoking Link", 
            "Prevalence in Lung Cancers"
        ],
        "Normal Lung Tissue": [
            "Diffuse throughout the lung lobe", 
            "Type I & II Pneumocytes", 
            "Delicate, thin, lace-like walls", 
            "Open, fully aerated, empty space", 
            "None", 
            "N/A (Healthy)"
        ],
        "Adenocarcinoma (ACA)": [
            "Peripheral parts of the lungs", 
            "Glandular epithelial cells", 
            "Tubular, glandular or acinar nests", 
            "Destroyed and filled with tumor nests", 
            "Moderate (Occurs in non-smokers too)", 
            "~40% (Most common)"
        ],
        "Squamous Cell Carcinoma (SCC)": [
            "Central bronchi, near major airways", 
            "Squamous epithelial cells", 
            "Crowded sheets, Keratin pearls", 
            "Completely blocked/obliterated", 
            "Extremely strong", 
            "~25-30%"
        ]
    }
    
    df_compare = pd.DataFrame(comparison_data)
    st.table(df_compare.set_index("Attribute"))


# PAGE 3: LOCAL SAMPLE GALLERY
elif st.session_state.page == "🖼️ Local Sample Gallery":
    st.markdown("<h1 class='main-title'>🖼️ Dataset Sample Gallery</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Browse histopathological slices provided in the database directory and send them directly to the Diagnosis Lab.</p>", unsafe_allow_html=True)
    
    st.write("Click on any category to view the dataset slide cuts. You can click the analysis button under an image to load it into the laboratory.")
    
    # Category tabs
    tab_aca, tab_n, tab_scc = st.tabs([
        "🧪 lung_aca Slices (Adenocarcinoma)", 
        "🍀 lung_n Slices (Normal)", 
        "🔥 lung_scc Slices (Squamous Carcinoma)"
    ])
    
    def render_gallery_grid(class_key, class_label):
        dir_path = os.path.join(BASE_DIR, class_key)
        if not os.path.exists(dir_path):
            st.error(f"Directory {class_key} not found.")
            return
            
        files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
        
        if not files:
            st.warning(f"No sample images found in {class_key}/")
            return
            
        # Draw a responsive grid of 5 columns
        cols = st.columns(5)
        for i, file in enumerate(files):
            col = cols[i % 5]
            img_path = os.path.join(dir_path, file)
            try:
                img = Image.open(img_path)
                with col:
                    st.image(img, width='stretch')
                    st.caption(f"**{file}**")
                    # Button to redirect to main page with this file selected
                    if st.button("Analyze slide", key=f"btn_{class_key}_{file}", width='stretch'):
                        st.session_state.selected_sample = {
                            "class": class_key,
                            "filename": file
                        }
                        st.session_state.page = "🔬 Diagnosis Lab"
                        st.rerun()
            except Exception as e:
                col.error(f"Error: {e}")

    with tab_aca:
        st.markdown("### Lung Adenocarcinoma Histology Samples")
        st.write("Observe the complex, crowded glandular clusters. The cells form ring-like nests, obliterating the normal air-filled spaces.")
        render_gallery_grid("lung_aca", "Adenocarcinoma")
        
    with tab_n:
        st.markdown("### Normal Lung Histology Samples")
        st.write("Notice the clear, open white pockets representing healthy alveoli. The thin pink lines represent alveolar septa with minimal cellularity.")
        render_gallery_grid("lung_n", "Normal Tissue")
        
    with tab_scc:
        st.markdown("### Lung Squamous Cell Carcinoma Histology Samples")
        st.write("Look for highly dense, sheets of irregular cells. Normal alveolar structures are entirely lost, replaced by thick tumor blocks.")
        render_gallery_grid("lung_scc", "Squamous Carcinoma")
