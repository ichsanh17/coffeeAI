import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="CoffeeAI - Klasifikasi Kopi",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS untuk tema coffee
st.markdown(
    """
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for coffee theme */
    :root {
        --coffee-dark: #3E2723;
        --coffee-medium: #5D4037;
        --coffee-light: #8D6E63;
        --coffee-cream: #EFEBE9;
        --coffee-white: #FAFAFA;
        --coffee-accent: #FF8F00;
        --coffee-text: #2E2E2E;
    }
    
    /* Hide streamlit style */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Custom styling */
    .stApp {
        background-color: var(--coffee-white);
        font-family: 'Poppins', sans-serif;
        color: var(--coffee-text);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--coffee-dark) 0%, var(--coffee-medium) 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(62, 39, 35, 0.1);
    }
    
    .main-title {
        color: var(--coffee-white) !important;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: var(--coffee-cream) !important;
        font-size: 1.2rem;
        text-align: center;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Card styling */
    .coffee-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(62, 39, 35, 0.08);
        border: 1px solid var(--coffee-cream);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .coffee-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(62, 39, 35, 0.12);
    }
    
    /* Upload area styling */
    .stFileUploader {
        border: 2px dashed var(--coffee-light);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: var(--coffee-cream);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--coffee-medium);
        background: white;
    }
    
    /* Fix all text colors */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: var(--coffee-text) !important;
    }
    
    .stMarkdown p {
        color: var(--coffee-text) !important;
    }
    
    .stFileUploader label {
        color: var(--coffee-medium) !important;
    }
    
    /* Result styling */
    .result-container {
        background: var(--coffee-cream);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid var(--coffee-accent);
    }
    
    .result-text {
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--coffee-dark) !important;
    }
    
    .confidence-text {
        font-size: 0.9rem;
        color: var(--coffee-medium) !important;
        margin-top: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid var(--coffee-accent);
        margin: 1rem 0;
    }
    
    .info-box p, .info-box strong {
        color: var(--coffee-text) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .main-subtitle {
            font-size: 1rem;
        }
        .coffee-card {
            padding: 1rem;
        }
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, var(--coffee-accent), var(--coffee-medium));
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Feature boxes */
    .feature-box {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(62, 39, 35, 0.05);
        margin: 1rem 0;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--coffee-dark) !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: var(--coffee-medium) !important;
        font-size: 0.9rem;
    }
    
    /* Fix spinner text */
    .stSpinner > div {
        color: var(--coffee-dark) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load models
@st.cache_resource
def load_models():
    """
    Load both models: TFLite for inference and Keras for GradCAM
    """
    # Load TFLite model for inference
    tflite_interpreter = tf.lite.Interpreter(model_path="EfficientNetV2B0_Model_Quantized_Float16 usk+17.tflite")
    tflite_interpreter.allocate_tensors()
    
    # Load Keras model for GradCAM
    keras_model = tf.keras.models.load_model("coffeeUSK_Model_EfficientNetV2B0.keras")
    
    return tflite_interpreter, keras_model


def predict_with_tflite(image_array, interpreter):
    """
    Make prediction using TFLite model
    """
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image to match input shape
    # input_data = image_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Assuming binary classification: [Normal, Defect]
    classes = ["Kopi Cacat", "Kopi Normal"]
    predicted_class_idx = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class_idx]
    predicted_class = classes[predicted_class_idx]
    print(output_data.shape)
    
    return predicted_class, confidence, output_data[0]


def get_last_conv_layer_name(model):
    """
    Get the name of the last convolutional layer
    """
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and not 'depthwise' in layer.name.lower():
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Create GradCAM heatmap for model explanation
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # The gradient of the output neuron (top predicted or chosen) with respect to the output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Mean of the gradient for each feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def create_gradcam_visualization(original_image, heatmap):
    """
    Create overlay of GradCAM heatmap on original image
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert original image to proper format if needed
    if original_image.shape[-1] == 3:  # RGB
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    else:
        original_bgr = original_image
    
    # Create overlay
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # Convert back to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    """
    # Resize image
    img = image.resize(target_size)

    # Convert to array
    img_array = img_to_array(img)

    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess for EfficientNet
    preprocessed_img = preprocess_input(img_array)

    
    return preprocessed_img, np.array(img)


# Main app
def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1 class="main-title">☕ CoffeeAI</h1>
        <p class="main-subtitle" style="color: var(--coffee-cream) !important">Sistem Klasifikasi Kopi Berbasis AI dengan EfficientNetV2B0</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load models
    tflite_interpreter, keras_model = load_models()
    
    # Get last conv layer name for GradCAM
    # last_conv_layer = get_last_conv_layer_name(keras_model)
    last_conv_layer = "top_conv"
    
    if last_conv_layer is None:
        st.error("Tidak dapat menemukan layer konvolusi terakhir untuk GradCAM")
        return

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # st.markdown('<div class="coffee-card">', unsafe_allow_html=True)
        st.markdown("### 📤 **Upload Gambar Kopi**")
        st.markdown("Unggah foto biji kopi untuk menganalisis kualitasnya")

        uploaded_file = st.file_uploader(
            "", type=["jpg", "jpeg", "png"], help="Format yang didukung: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_container_width =True)
            
            # Automatic classification after upload
            with st.spinner("Menganalisis kopi... ☕"):
                # Preprocess image
                image_array, original_image = preprocess_image(image)
                
                # Make prediction with TFLite
                prediction, confidence, predictions = predict_with_tflite(image_array, tflite_interpreter)
                
                # Generate GradCAM with Keras model
                pred_index = np.argmax(predictions)
                heatmap = make_gradcam_heatmap(image_array, keras_model, last_conv_layer, pred_index)
                
                # Store results in session state
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.heatmap = heatmap
                st.session_state.original_image = original_image
                st.session_state.predictions = predictions

        st.markdown("</div>", unsafe_allow_html=True)

        

    with col2:
        # st.markdown('<div class="coffee-card">', unsafe_allow_html=True)
        st.markdown("### 📊 **Hasil Analisis**")

        if hasattr(st.session_state, "prediction"):
            # Display prediction result
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            predictions = st.session_state.predictions

            # Result styling based on prediction
            result_color = "#4CAF50" if prediction == "Kopi Normal" else "#FF5722"
            # result_emoji = "✅" if prediction == "Kopi Normal" else "❌"

            st.markdown(
                f"""
            <div class="result-container">
                <div class="result-text">
                    <strong>Klasifikasi: {prediction}</strong>
                </div>
                <div class="confidence-text">
                    Tingkat Kepercayaan: {confidence:.2%}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Progress bar for confidence
            st.progress(float(confidence))
            
            st.markdown("#### 📈 Detail Prediksi")
            # Data
            classes = ["Kopi Cacat", "Kopi Normal"]
           

            # Styling menggunakan CSS
            st.markdown(
                """
                <style>
                    tbody tr:nth-child(odd) {
                        background-color: #D2691E;  /* Warna coklat kopi */
                        color: var(--coffee-cream) !important;
                    }
                    tbody tr:nth-child(even) {
                        background-color: #8B4513;  /* Warna coklat lebih gelap */
                        color: var(--coffee-cream) !important;
                    }
                    th {
                        background-color: #3E2723;  /* Coklat gelap untuk header */
                        color: var(--coffee-cream) !important;
                        
                    }
                    td {
                        
                        color: var(--coffee-cream) !important;
                    }
                </style>
                """, unsafe_allow_html=True
            )

            # Tabel prediksi
            pred_df = {
                "Kelas": classes,
                "Probabilitas": [f"{pred:.2%}" for pred in predictions]
            }
            st.table(pred_df)


            # GradCAM visualization
            st.markdown("#### 🔍 **Analisis XAI dengan GradCAM**")
            st.markdown("Visualisasi area yang diperhatikan model saat membuat prediksi:")

            if hasattr(st.session_state, "heatmap"):
                # Create GradCAM visualization
                gradcam_image = create_gradcam_visualization(
                    st.session_state.original_image, 
                    st.session_state.heatmap
                )

                # Display original and GradCAM side by side
                grad_col1, grad_col2 = st.columns(2)

                with grad_col1:
                    st.image(
                        st.session_state.original_image,
                        caption="Gambar Asli",
                        use_container_width =True,
                    )

                with grad_col2:
                    st.image(
                        gradcam_image,
                        caption="GradCAM Visualization",
                        use_container_width =True,
                    )

                # Explanation
                st.markdown(
                    """
                <div class="info-box">
                    <strong>💡 Penjelasan GradCAM:</strong><br>
                    Area berwarna merah menunjukkan bagian gambar yang paling berpengaruh 
                    dalam keputusan model. Area berwarna biru menunjukkan bagian yang 
                    kurang berpengaruh dalam proses klasifikasi.
                </div>
                """,
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                """
            <div class="info-box">
                <p>📷 Silakan unggah gambar kopi untuk memulai analisis.</p>
                <p>Model akan otomatis mengklasifikasikan apakah kopi tersebut normal atau memiliki cacat.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Additional info
        st.markdown("### 📋 **Informasi Model**")
        st.markdown(
            """
        <div class="info-box">
            <strong>Model Inference:</strong> EfficientNetV2B0 (TFLite)<br>
            <strong>Model GradCAM:</strong> EfficientNetV2B0 (Keras)<br>
            <strong>Kelas:</strong> Kopi Normal, Kopi Cacat<br>
            <strong>Input:</strong> Gambar RGB 224x224 piksel<br>
            <strong>XAI:</strong> GradCAM untuk interpretabilitas
        </div>
        """,
            unsafe_allow_html=True,
        )


# # Footer
# st.markdown("---")
# st.markdown(
#     """
# <div style="text-align: center; color: var(--coffee-medium); font-size: 0.9rem; padding: 1rem;">
#     ☕ Dibuat dengan ❤️ untuk kualitas kopi yang lebih baik
# </div>
# """,
#     unsafe_allow_html=True,
# )

if __name__ == "__main__":
    main()