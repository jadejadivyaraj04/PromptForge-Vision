# PromptForge Vision API

A powerful, high-performance FastAPI microservice that utilizes a **Dual-Model Generative AI Pipeline** to dynamically enhance user prompts and generate stunning AI imagery using Google's Gemini models.

This project demonstrates expertise in:
- API Development with **FastAPI**
- Multi-step LLM pipelines (Text-to-Text -> Text-to-Image)
- Prompt Engineering and Context Optimization
- Stateless Architecture & Security (API Keys injected dynamically via Headers)

## 🚀 Architecture: The Dual-Model Pipeline

Instead of passing user input directly to an image generator (which often results in bland images), this API employs a two-step "Prompt Enhancer" pipeline:

1. **Step 1: The Enhancer (Gemini 2.5 Flash)**
   The API takes a simple `title` and `description` from the user and automatically rewrites it using a high-speed text model. It acts as an expert Prompt Engineer, adding cinematic lighting, camera angles, and rendering styles.
2. **Step 2: The Generator (Gemini 3 Pro Image Preview)**
   The enhanced "Master Prompt" is then fed into the image generation model, resulting in high-fidelity, professional-grade artwork.

## 🛠 Tech Stack
- **Python 3.10+**
- **FastAPI** & **Uvicorn**
- **Google GenAI SDK** (Gemini 2.5 Flash & Gemini 3 Pro)
- **Streamlit** (Included for local UI testing)

## 🔌 API Reference

### `POST /generate`

Dynamically generates a base64 encoded image from a text concept.

**Headers Required:**
```http
Content-Type: application/json
x-gemini-api-key: YOUR_GEMINI_API_KEY
```

**Request Body:**
```json
{
  "title": "A flying car",
  "description": "Cyberpunk city, neon lights reflecting on wet streets, nighttime"
}
```

**Response:**
```json
{
  "enhanced_prompt": "A cinematic, hyper-realistic 8k render of a sleek flying car soaring above a futuristic cyberpunk city. Neon pink and cyan lights reflect brilliantly off the wet, rain-slicked streets below. Dark, moody atmosphere, volumetric lighting, unreal engine 5 style.",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## 💻 Local Development

1. Clone the repository
2. Create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the API: `uvicorn api:app --reload`
5. Run the UI Dashboard: `streamlit run app.py`
