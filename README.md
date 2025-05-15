# Optical Character Recognition (OCR) Streamlit App


A web application built with Streamlit to demonstrate Optical Character Recognition (OCR) capabilities. Users can upload images, apply various preprocessing techniques, and extract text using the Tesseract OCR engine. The app visualizes detected text regions with bounding boxes, similar to the project's demonstration goals.

## ✨ Features

*   **Image Upload:** Supports uploading common image formats (PNG, JPG, JPEG).
*   **Interactive Preprocessing:**
    *   Grayscale conversion
    *   Gaussian blur with adjustable kernel size
    *   Multiple binarization/thresholding methods (Otsu, Adaptive Mean, Adaptive Gaussian)
    *   Contrast Limited Adaptive Histogram Equalization (CLAHE)
*   **Text Detection & Recognition:**
    *   Utilizes Tesseract OCR engine.
    *   Adjustable language for OCR (e.g., 'eng', 'urd').
*   **Visualization:**
    *   Displays original and preprocessed images.
    *   Overlays bounding boxes on detected words in the original image..
*   **Text Output:**
    *   Displays the full extracted text.
    *   Shows a table of detected words with confidence scores and bounding box coordinates.
*   **Responsive UI:** Built with Streamlit for a clean and interactive user experience.

##  Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python:** Version 3.8 or higher. You can download it from [python.org](https://www.python.org/downloads/).
2.  **Tesseract OCR Engine:** This is crucial for the OCR functionality.
    *   **Windows:** Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
        *   **Important:** During installation, make sure to select additional language data (e.g., for English - `eng.traineddata`) and add Tesseract to your system's PATH environment variable.
    *   **Linux (Ubuntu/Debian):**
        ```bash
        sudo apt-get update
        sudo apt-get install tesseract-ocr
        sudo apt-get install tesseract-ocr-eng # For English language data
        # For other languages, e.g., Urdu: sudo apt-get install tesseract-ocr-urd
        ```
    *   **macOS (using Homebrew):**
        ```bash
        brew install tesseract
        brew install tesseract-lang # Installs all language data
        ```
    *   **Verify Tesseract Installation:** Open your terminal or command prompt and type `tesseract --version`. You should see the Tesseract version information.

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ABDULLAHAZHERCH/Optical-Character-Recognition.git
    cd Optical-Character-Recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    *   Activate the virtual environment:
        *   Windows: `venv\Scripts\activate`
        *   macOS/Linux: `source venv/bin/activate`

3.  **Install Python dependencies:**
    The required libraries are listed in `app.py`. You can install them directly:
    ```bash
    pip install streamlit opencv-python-headless pytesseract Pillow numpy
    ```
    Alternatively, create a `requirements.txt` file with the following content:
    ```
    streamlit
    opencv-python-headless
    pytesseract
    Pillow
    numpy
    ```
    And then install using:
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Windows Specific) Tesseract Path Configuration (if needed):**
    If `pytesseract` cannot find your Tesseract installation (even if it's in PATH), you might need to specify the path to `tesseract.exe` at the beginning of your `app.py` script. Uncomment and update the following line in `app.py`:
    ```python
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```

## 🏃 Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py
