import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tempfile
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

MODELS_DIR = "modelsv3"
RUNTIME_DIR = "runtime_files"
SAMPLE_IMAGES_DIR = "sample_images"
MODEL_FILES = {
    "EfficientNetB0": "efficientnetb0_model.h5",
    "MobileNetV2": "mobilenetv2_model.h5",
    "ResNet50": "resnet50_model.h5",
}

try:
    os.makedirs(RUNTIME_DIR, exist_ok=True)
except OSError:
    RUNTIME_DIR = tempfile.mkdtemp(prefix="pathonix_runtime_")


def is_git_lfs_pointer(path):
    try:
        with open(path, "rb") as file:
            return file.read(64).startswith(b"version https://git-lfs.github.com")
    except OSError:
        return False


def validate_model_files():
    missing_or_invalid = []
    for model_name, file_name in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, file_name)
        if not os.path.exists(path):
            missing_or_invalid.append(f"{model_name}: missing `{path}`")
        elif is_git_lfs_pointer(path):
            missing_or_invalid.append(
                f"{model_name}: `{path}` is a Git LFS pointer, not the real model file"
            )
        elif os.path.getsize(path) < 1_000_000:
            missing_or_invalid.append(
                f"{model_name}: `{path}` is too small to be a trained Keras model"
            )
    return missing_or_invalid


@st.cache_resource
def load_models():
    problems = validate_model_files()
    if problems:
        problem_text = "\n".join(f"- {problem}" for problem in problems)
        raise RuntimeError(
            "The trained model files are not available locally.\n\n"
            f"{problem_text}\n\n"
            "Fix: install Git LFS and run `git lfs pull`, or copy the three real "
            "`.h5` files into the `modelsv3/` folder."
        )
    return {
        model_name: tf.keras.models.load_model(os.path.join(MODELS_DIR, file_name))
        for model_name, file_name in MODEL_FILES.items()
    }

# Grad-CAM functions
def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list):
            preds = preds[0]  # Convert list to tensor
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.math.reduce_max(heatmap)
    if float(max_value.numpy()) == 0.0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    heatmap = heatmap / max_value
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = load_img(img_path)
    img = img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return cam_path

def ensemble_predictions(models, input_data):
    predictions = [model.predict(input_data) for model in models]
    averaged_predictions = np.mean(predictions, axis=0)
    return averaged_predictions

# Streamlit UI
st.set_page_config(
    page_title="PathoNix 🩺🔬",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("PathoNix: Deep Learning for Enhanced Pathology Diagnostics 🩺🔬")
st.write("""
    Welcome to PathoNix, an advanced tool for classifying histopathological images 
    using state-of-the-art deep learning models. With our platform, you can:
    
    - **Upload histopathological images** to classify them as cancerous or non-cancerous.
    - **Visualize Grad-CAM heatmaps** to understand the model's decision-making process.
    
    Simply upload an image to get started! 🚀
""")

try:
    models = load_models()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

efficientnetb0_model = models["EfficientNetB0"]
mobilenetv2_model = models["MobileNetV2"]
resnet50_model = models["ResNet50"]

# Radio button for selecting input type
input_type = st.radio(
    "Choose input type:",
    ("Upload an image", "Select a sample test image")
)

img_path = None

if input_type == "Upload an image":
    uploaded_file = st.file_uploader("Choose a histopathological image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_path = os.path.join(RUNTIME_DIR, os.path.basename(uploaded_file.name))
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Image uploaded successfully! Please click the **Diagnose** button to proceed.")
elif input_type == "Select a sample test image":
    sample_images = sorted(
        file_name
        for file_name in os.listdir(SAMPLE_IMAGES_DIR)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if sample_images:
        selected_image = st.selectbox("Choose a sample test image", sample_images)
        img_path = os.path.join(SAMPLE_IMAGES_DIR, selected_image)
        st.write("Sample image selected successfully! Please click the **Diagnose** button to proceed.")
    else:
        st.warning("No sample images were found in the `sample_images/` folder.")

# Add a submit button
if img_path and st.button("Diagnose 🩺"):
    st.write("Classifying... Please wait ⏳")

    # Show a step-by-step spinner
    with st.spinner("Processing..."):
        progress_text = st.empty()
        progress_text.write("Step 1/5: Preparing image...")
        # Prepare image
        img_array = get_img_array(img_path, size=(224, 224))

        progress_text.write("Step 2/5: Generating Grad-CAM for EfficientNetB0...")
        # Predict and visualize Grad-CAM for EfficientNetB0
        efficientnetb0_heatmap = make_gradcam_heatmap(
            img_array, efficientnetb0_model, "top_conv"
        )
        efficientnetb0_cam_path = os.path.join(RUNTIME_DIR, "efficientnetb0_cam.jpg")
        display_gradcam(img_path, efficientnetb0_heatmap, cam_path=efficientnetb0_cam_path)

        progress_text.write("Step 3/5: Generating Grad-CAM for MobileNetV2...")
        # Predict and visualize Grad-CAM for MobileNetV2
        mobilenetv2_heatmap = make_gradcam_heatmap(
            img_array, mobilenetv2_model, "Conv_1"
        )
        mobilenetv2_cam_path = os.path.join(RUNTIME_DIR, "mobilenetv2_cam.jpg")
        display_gradcam(img_path, mobilenetv2_heatmap, cam_path=mobilenetv2_cam_path)

        progress_text.write("Step 4/5: Generating Grad-CAM for ResNet50...")
        # Predict and visualize Grad-CAM for ResNet50
        resnet50_heatmap = make_gradcam_heatmap(
            img_array, resnet50_model, "conv5_block3_out"
        )
        resnet50_cam_path = os.path.join(RUNTIME_DIR, "resnet50_cam.jpg")
        display_gradcam(img_path, resnet50_heatmap, cam_path=resnet50_cam_path)

        progress_text.write("Step 5/5: Performing ensemble prediction...")
        # Ensemble prediction
        ensemble_pred = ensemble_predictions(
            [efficientnetb0_model, mobilenetv2_model, resnet50_model], img_array
        )

    st.success("Processing complete!")

    # Display image comparison using image_comparison slider
    st.write("### Visualizations 🔍")
    image_comparison(
        img1=img_path,
        img2=efficientnetb0_cam_path,
        label1="Original Image",
        label2="EfficientNetB0 Grad-CAM",
    )
    image_comparison(
        img1=img_path,
        img2=mobilenetv2_cam_path,
        label1="Original Image",
        label2="MobileNetV2 Grad-CAM",
    )
    image_comparison(
        img1=img_path,
        img2=resnet50_cam_path,
        label1="Original Image",
        label2="ResNet50 Grad-CAM",
    )

    st.write("### Ensemble Prediction 📊")
    # Assuming a threshold of 0.5 for binary classification
    ensemble_score = float(np.squeeze(ensemble_pred))
    st.write(f"Ensemble score: **{ensemble_score:.3f}**")
    if ensemble_score >= 0.5:
        st.write("The model predicts: **Non-Cancer** 💊")
    else:
        st.write("The model predicts: **Cancer** 🧫")

# Sidebar with logo and about information
st.sidebar.image("./Images/living-tissue.png", use_column_width=True)
st.sidebar.title("About PathoNix 🩺🔬")
st.sidebar.info(
    """
    **PathoNix** is an advanced tool for classifying histopathological images using state-of-the-art deep learning models.
    
    **Features:**
    - Utilizes EfficientNetB0, MobileNetV2, and ResNet50 models.
    - Provides Grad-CAM visualizations for model interpretability.
    - Uses ensemble predictions for improved accuracy.

    **Developed by:**
    - Vishnu Mukku

    **Contact:**
    - Email: vishnu.mukku.2020@gmail.com
    
    **License:**
    This project is licensed under the MIT License.
    """
)
