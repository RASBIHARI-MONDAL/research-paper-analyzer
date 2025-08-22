import streamlit as st
import os
import pdfplumber
from PIL import Image
import pytesseract
import google.generativeai as genai

# ==== CONFIGURE GOOGLE GEMINI ====
API_KEY = "enter your api key"  # <----- Replace with your actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


# ==== TEXT EXTRACTION FUNCTIONS ====

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text


# ==== ANALYSIS FUNCTION ====

def analyze_text(text):
    prompt = f"""
You are an expert AI assistant for academic research.

Given the following research paper content, perform these tasks:
1. Provide a comprehensive summary of the paper in approximately 500 words.
2. List the 10 most important keywords.
3. Extract the 'Results' section, if available.
4. Extract the 'Experiments' or 'Methods' section, if available.

Return the output in this format using clear Markdown:

---
📄 **Summary (500 words)**:
<detailed summary>

🧠 **Keywords**:
- keyword1
- keyword2
- keyword3
- keyword4
- keyword5
- keyword6
- keyword7
- keyword8
- keyword9
- keyword10

📊 **Results**:
<results section>

🔬 **Experiments / Methods**:
<experiments or methods section>

Research Paper Content:
{text}
"""

    response = model.generate_content(prompt)
    return response.text


# ==== STREAMLIT APP UI ====

st.title("📄 AI Scientific Research Analyzer")
st.write("Upload a `.pdf`, `.jpg`, or `.jpeg` research paper to get a detailed 500-word summary, key terms, results, and experiments.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    st.info(f"Processing `{uploaded_file.name}`...")

    try:
        if file_ext == ".pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        elif file_ext in [".jpg", ".jpeg"]:
            raw_text = extract_text_from_image(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        if not raw_text.strip():
            st.warning("No readable text found in the file.")
        else:
            with st.spinner("Generating detailed summary with Gemini 1.5 Flash..."):
                result = analyze_text(raw_text)

            st.success("Analysis complete!")
            st.markdown("### 📄 Summary, Keywords, Results & Experiments")
            st.markdown(result)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

