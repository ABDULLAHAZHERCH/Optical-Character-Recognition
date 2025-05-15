import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

# --- Tesseract Configuration (Update if necessary) ---
# On Windows, you might need to explicitly point to the tesseract executable
# For example:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, if tesseract is in PATH, this is often not needed.

# --- Helper Functions ---
def preprocess_image(image_cv, grayscale, blur, threshold_type, clahe_apply):
    """Applies selected preprocessing steps to the image."""
    processed_img = image_cv.copy()

    if grayscale:
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    if blur > 0:
        # Ensure blur kernel is odd
        blur_kernel = blur if blur % 2 == 1 else blur + 1
        if grayscale:
            processed_img = cv2.GaussianBlur(processed_img, (blur_kernel, blur_kernel), 0)
        else: # Apply to each channel if not grayscale yet
            b, g, r = cv2.split(processed_img)
            b = cv2.GaussianBlur(b, (blur_kernel, blur_kernel), 0)
            g = cv2.GaussianBlur(g, (blur_kernel, blur_kernel), 0)
            r = cv2.GaussianBlur(r, (blur_kernel, blur_kernel), 0)
            processed_img = cv2.merge([b, g, r])


    if threshold_type != "None":
        if not grayscale: # Thresholding typically works best on grayscale
            img_for_thresh = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            img_for_thresh = processed_img.copy()

        if threshold_type == "Otsu":
            _, processed_img = cv2.threshold(img_for_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == "Adaptive Mean":
            processed_img = cv2.adaptiveThreshold(img_for_thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == "Adaptive Gaussian":
            processed_img = cv2.adaptiveThreshold(img_for_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
        # If thresholding was applied, convert back to BGR if original was color and grayscale wasn't selected
        # This helps if further color-based processing or display is needed.
        # However, for Tesseract, grayscale is often preferred.
        if len(processed_img.shape) == 2: # if it's a 2D (grayscale) image
             processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)


    if clahe_apply:
        if grayscale or len(processed_img.shape) == 2: # if already grayscale or became grayscale
            if len(processed_img.shape) == 3: # if it was converted back to BGR after thresholding
                 img_for_clahe = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            else:
                 img_for_clahe = processed_img
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_img_clahe = clahe.apply(img_for_clahe)
            processed_img = cv2.cvtColor(processed_img_clahe, cv2.COLOR_GRAY2BGR) # convert back for display
        else: # Apply CLAHE to L channel of LAB color space for color images
            lab = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            processed_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return processed_img


def perform_ocr(image_cv, lang='eng'):
    """Performs OCR using Tesseract and returns text and data (for bounding boxes)."""
    try:
        # For better bounding boxes, use image_to_data
        # Tesseract works best with Grayscale images for OCR
        if len(image_cv.shape) == 3:
            ocr_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            ocr_image = image_cv

        # Sometimes, a bit of thresholding helps Tesseract even after other preprocessing
        # _, ocr_image = cv2.threshold(ocr_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--oem 3 --psm 6' # Page Segmentation Mode 6: Assume a single uniform block of text.
        text_data = pytesseract.image_to_data(ocr_image, lang=lang, output_type=pytesseract.Output.DICT, config=custom_config)
        full_text = pytesseract.image_to_string(ocr_image, lang=lang, config=custom_config)
        return full_text, text_data
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract is not installed or not in your PATH. Please install Tesseract OCR.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during OCR: {e}")
        return None, None

def draw_boxes_on_image(image_pil, text_data, label_words=False):
    """Draws bounding boxes on the image based on Tesseract data."""
    if text_data is None:
        return image_pil

    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    n_boxes = len(text_data['level'])

    default_font_size = 15
    try:
        font = ImageFont.truetype("arial.ttf", default_font_size)
    except IOError:
        font = ImageFont.load_default()


    for i in range(n_boxes):
        # We are interested in word-level boxes (level 5) or line-level (level 4)
        # For your example, word level is more appropriate.
        if int(text_data['conf'][i]) > 30 and text_data['level'][i] == 5: # Confidence > 30, level 5 is word
            (x, y, w, h) = (text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i])
            word_text = text_data['text'][i].strip()

            if word_text: # Only draw if there's text
                draw.rectangle([(x, y), (x + w, y + h)], outline="purple", width=2)
                if label_words:
                    # Position text above the box
                    text_x = x
                    text_y = y - default_font_size - 2 # 2 pixels padding
                    if text_y < 0: # If text goes off screen on top, put it inside
                        text_y = y + 2

                    # Simple background for text for better readability
                    text_bbox = draw.textbbox((text_x, text_y), word_text, font=font)
                    draw.rectangle(text_bbox, fill="purple")
                    draw.text((text_x, text_y), word_text, fill="white", font=font)
    return img_draw

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Optical Character Recognition")

st.title("📄 Optical Character Recognition (OCR) System")
st.markdown("""
This application demonstrates OCR capabilities.
""")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

st.sidebar.subheader("Image Preprocessing")
apply_grayscale = st.sidebar.checkbox("Convert to Grayscale", value=True)
apply_blur = st.sidebar.slider("Gaussian Blur Kernel Size (0 to disable)", 0, 21, 0, step=2)
thresholding_option = st.sidebar.selectbox(
    "Binarization/Thresholding",
    ["None", "Otsu", "Adaptive Mean", "Adaptive Gaussian"],
    index=0
)
apply_clahe = st.sidebar.checkbox("Apply CLAHE (Contrast Enhancement)", value=False)

st.sidebar.subheader("OCR Settings")
ocr_language = st.sidebar.text_input("Tesseract Language (e.g., 'eng', 'urd')", value="eng")
label_detected_words = st.sidebar.checkbox("Label Detected Words on Image", value=True, help="Show recognized word above its bounding box, similar to example.")


# --- Main Content Area ---
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    pil_image = Image.open(uploaded_file).convert("RGB")
    opencv_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(pil_image, use_column_width=True)

    # Preprocess the image
    preprocessed_cv_image = preprocess_image(opencv_image, apply_grayscale, apply_blur, thresholding_option, apply_clahe)
    preprocessed_pil_image = Image.fromarray(cv2.cvtColor(preprocessed_cv_image, cv2.COLOR_BGR2RGB))

    with col2:
        st.subheader("✨ Preprocessed Image")
        st.image(preprocessed_pil_image, use_column_width=True,
                 caption="This image will be fed to Tesseract.")

    st.markdown("---")
    st.subheader("🔍 OCR Results")

    if st.button("👁️‍🗨️ Perform OCR and Detect Text", type="primary"):
        with st.spinner("Recognizing text... This might take a moment."):
            extracted_text, text_data_dict = perform_ocr(preprocessed_cv_image, lang=ocr_language)

            if extracted_text is not None and text_data_dict is not None:
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("#### 🔡 Detected Text Blocks")
                    # Create an image with bounding boxes for display
                    image_with_boxes_pil = draw_boxes_on_image(pil_image.copy(), text_data_dict, label_words=label_detected_words)
                    st.image(image_with_boxes_pil, use_column_width=True,
                             caption="Detected text regions on original image.")

                with col_res2:
                    st.markdown("#### 📝 Extracted Text")
                    st.text_area("Full recognized text:", extracted_text, height=300)

                    st.markdown("#### 📊 Word-Level Data (from Tesseract)")
                    if text_data_dict:
                        words_info = []
                        for i in range(len(text_data_dict['text'])):
                            if text_data_dict['text'][i].strip() and text_data_dict['level'][i] == 5 : # level 5 is word
                                words_info.append({
                                    "Word": text_data_dict['text'][i],
                                    "Conf.": text_data_dict['conf'][i],
                                    "Box (x,y,w,h)": f"({text_data_dict['left'][i]}, {text_data_dict['top'][i]}, {text_data_dict['width'][i]}, {text_data_dict['height'][i]})"
                                })
                        if words_info:
                            st.dataframe(words_info)
                        else:
                            st.info("No distinct words found with sufficient confidence.")
                    else:
                        st.warning("No text data dictionary returned from OCR.")

            elif extracted_text is None and text_data_dict is None:
                # Error already handled by perform_ocr
                pass
            else:
                st.warning("OCR did not return expected data.")
else:
    st.info("👈 Upload an image using the sidebar to get started!")

st.markdown("---")
# --- Displaying Proposal Information ---
with st.expander("ℹ️ About This Project"):
    st.header("Project Objectives")
    st.markdown("""
    The primary objectives of this project are:
    1. To develop a robust image preprocessing pipeline using OpenCV to enhance text visibility.
    2. To implement accurate text detection and segmentation methods for isolating text regions.
    3. To utilize Tesseract OCR engine for character recognition with optimized configuration.
    4. To create a real-time text extraction system capable of processing video streams (Note: This demo focuses on static images).
    5. To build an interactive user interface demonstrating the system's capabilities (This Streamlit app is a part of that).
    6. To evaluate the performance of different preprocessing algorithms and OCR approaches.
    7. To implement post-processing techniques for improving recognition accuracy.
    """)

    st.header("System Architecture Modules")
    st.markdown("""
    The proposed OCR system follows a modular pipeline architecture:
    1.  **Image Acquisition Module:** Handles input (file, camera, video). *This demo uses file upload.*
    2.  **Preprocessing Module:** Enhances image quality. *Implemented with OpenCV options in the sidebar.*
    3.  **Text Detection Module:** Identifies text regions. *Tesseract `image_to_data` provides this.*
    4.  **Text Recognition Module:** Employs Tesseract. *Implemented.*
    5.  **Post-processing Module:** Refines results. *(Future work for this demo)*
    6.  **User Interface Module:** Provides interaction. *This Streamlit app.*
    """)

    st.header("Technologies Used in this Demo")
    st.markdown("""
    - **Streamlit:** For the web application user interface.
    - **OpenCV:** For image processing and manipulation.
    - **Pytesseract:** Python wrapper for Google's Tesseract OCR Engine.
    - **Pillow (PIL):** For image handling, drawing.
    - **NumPy:** For numerical operations with images.
    """)

st.markdown("<hr><p style='text-align:center;'>CHAUDHARY & SHAHMIR</p>", unsafe_allow_html=True)