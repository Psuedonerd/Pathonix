# PathoNix 🩺🔬

**PathoNix** is an advanced tool for classifying histopathological images using state-of-the-art deep learning models. With PathoNix, you can upload histopathological images to classify them as cancerous or non-cancerous, and visualize Grad-CAM heatmaps to understand the model's decision-making process.

## Features

- **Upload histopathological images** to classify them as cancerous or non-cancerous.
- **Visualize Grad-CAM heatmaps** to understand the model's decision-making process.
- **Ensemble predictions** for improved accuracy using EfficientNetB0, MobileNetV2, and ResNet50 models.
- **Interactive image comparison** for original and Grad-CAM images.

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your-username/pathonix.git
cd pathonix
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

```
.
├── Images
│   └── living-tissue.png
├── modelsv3
│   ├── efficientnetb0_model.h5
│   ├── mobilenetv2_model.h5
│   └── resnet50_model.h5
├── runtime_files
├── temp
├── .gitignore
├── Ovarian_Cancer_Detection_Preprocessing_and_Augmentation.ipynb
├── packages.txt
├── Readme.md
├── requirements.txt
├── StreamlitApp.py
└── config.toml
```

### Running the Application

1. **Start the Streamlit app:**

```bash
streamlit run StreamlitApp.py
```

2. **Open your web browser and navigate to:**

```
http://localhost:8501
```

## Usage

1. **Upload an Image:**
   - Choose a histopathological image in JPG, JPEG, or PNG format.

2. **Classify and Visualize:**
   - The app will classify the image and display Grad-CAM heatmaps for EfficientNetB0, MobileNetV2, and ResNet50 models.
   - You can compare the original image with the Grad-CAM heatmaps using an interactive slider.

3. **Ensemble Prediction:**
   - The app will perform an ensemble prediction and display the result (Cancer or Non-Cancer).

## Configuration

### Streamlit Theme Configuration

The app uses a custom theme. You can configure the theme by modifying the `config.toml` file.

```toml
[theme]
primaryColor = "#A93226"        # Red
backgroundColor = "#FEF9E7"     # Cream
secondaryBackgroundColor = "#F5CBA7" # Light Brown
textColor = "#000000"           # Black
accentColor = "#7D3C98"         # Purple
font = "sans serif"

[browser]
gatherUsageStats = false
```

## Contact

**Developed by:**
- Your Name
- Team Members (if any)
- Organization/Institution (if applicable)

**Contact:**
- Email: your-email@example.com
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/your-profile/)
- GitHub: [Your GitHub](https://github.com/your-profile/)

## License

This project is licensed under the MIT License.




## References:

1. [Grad CAM in Deep Learning](https://www.analyticsvidhya.com/blog/2023/12/grad-cam-in-deep-learning/)

The journey of Image Classification from Pixels to Perfection has been such revolutionary, transforming the impossible into reality. 📸

Here's how a few notable groundbreaking models have evolved over time, each bringing us closer to perfection:

2. 2015: ResNet

- Developed by Kaiming He, Xiangyu Zhang, [**Shaoqing Ren**](https://www.linkedin.com/feed/update/urn:li:activity:7213904086025351169/#), and Jian Sun, ResNet introduced residual connections to address the vanishing gradient problem. 🔄

🔹Advancement: Enabled training of extremely deep networks (up to 152 layers), achieving a top-5 error rate of 3.6%. 🌟

3. 2019: EfficientNet

- Created by [**Mingxing Tan**](https://www.linkedin.com/feed/update/urn:li:activity:7213904086025351169/#) and [**Quoc V. Le**](https://www.linkedin.com/feed/update/urn:li:activity:7213904086025351169/#), EfficientNet used a new scaling method that uniformly scales all dimensions. 📈

🔹Advancement: Achieved state-of-the-art accuracy with fewer parameters and FLOPs, showing that careful scaling can improve performance. 🏅


