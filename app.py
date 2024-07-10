import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import shutil
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

# Load models
models_dir = "modelsv3"
efficientnetb0_model = tf.keras.models.load_model(
    os.path.join(models_dir, "efficientnetb0_model.h5")
)
mobilenetv2_model = tf.keras.models.load_model(
    os.path.join(models_dir, "mobilenetv2_model.h5")
)
resnet50_model = tf.keras.models.load_model(
    os.path.join(models_dir, "resnet50_model.h5")
)

# Ensure runtime_files directory exists
runtime_dir = "runtime_files"
os.makedirs(runtime_dir, exist_ok=True)

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
        print(f"Shape of preds: {preds.shape}, type of preds: {type(preds)}")  # Debugging statement
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        print(f"Predicted index: {pred_index}, type of pred_index: {type(pred_index)}")  # Debugging statement
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
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

uploaded_file = st.file_uploader("Choose a histopathological image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = os.path.join(runtime_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

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
        efficientnetb0_cam_path = os.path.join(runtime_dir, "efficientnetb0_cam.jpg")
        display_gradcam(img_path, efficientnetb0_heatmap, cam_path=efficientnetb0_cam_path)

        progress_text.write("Step 3/5: Generating Grad-CAM for MobileNetV2...")
        # Predict and visualize Grad-CAM for MobileNetV2
        mobilenetv2_heatmap = make_gradcam_heatmap(
            img_array, mobilenetv2_model, "Conv_1"
        )
        mobilenetv2_cam_path = os.path.join(runtime_dir, "mobilenetv2_cam.jpg")
        display_gradcam(img_path, mobilenetv2_heatmap, cam_path=mobilenetv2_cam_path)

        progress_text.write("Step 4/5: Generating Grad-CAM for ResNet50...")
        # Predict and visualize Grad-CAM for ResNet50
        resnet50_heatmap = make_gradcam_heatmap(
            img_array, resnet50_model, "conv5_block3_out"
        )
        resnet50_cam_path = os.path.join(runtime_dir, "resnet50_cam.jpg")
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
    if ensemble_pred >= 0.5:
        st.write("The model predicts: **Non-Cancer** 💊")
    else:
        st.write("The model predicts: **Cancer** 🧫")

    # Clean up runtime_files directory
    shutil.rmtree(runtime_dir)
    os.makedirs(runtime_dir, exist_ok=True)

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
    - Rising Sophomore at Avon High School

    **Contact:**
    - Email: vishnu.mukku.2020@gmail.com
    - LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/your-profile/)
    - GitHub: [Your GitHub](https://github.com/your-profile/)
    
    **License:**
    This project is licensed under the MIT License.
    """
)
