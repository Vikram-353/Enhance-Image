import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Page setup
st.set_page_config(page_title="Image Enhancement Suite", layout="wide")
st.title("📸 Image Enhancement Suite")
st.markdown("Upload an image to apply one of the three enhancement techniques: **CLAHE**, **Unsharp Masking**, or **Gamma Correction**.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Gamma adjustment function
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Select enhancement method
    method = st.selectbox("Select enhancement method", ["CLAHE", "Unsharp Masking", "Gamma Correction"])

    if method == "CLAHE":
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        caption = "Enhanced Image (CLAHE)"

    elif method == "Unsharp Masking":
        blurred = cv2.GaussianBlur(img_bgr, (9, 9), 10.0)
        sharpened = cv2.addWeighted(img_bgr, 1.5, blurred, -0.9, 5)
        enhanced_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        caption = "Sharpened Image (Unsharp Masking)"

    elif method == "Gamma Correction":
        col1, col2, col3 = st.columns(3)
        with col2:
            gamma_val = st.slider("Gamma Value", 0.1, 3.0, 1.0, 0.1)
        gamma_corrected = adjust_gamma(img_bgr, gamma=gamma_val)
        enhanced_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
        caption = f"Gamma Corrected Image (γ = {gamma_val})"

    # Display side-by-side
    st.subheader("Result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Original Image", use_container_width=True)
    with col2:
        st.image(enhanced_rgb, caption=caption, use_container_width=True)

    # Download the enhanced image
    result_image = Image.fromarray(enhanced_rgb)
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        label="📥 Download Enhanced Image",
        data=buf,
        file_name="enhanced_image.png",
        mime="image/png"
    )

else:
    st.info("📂 Please upload a valid image file.")
