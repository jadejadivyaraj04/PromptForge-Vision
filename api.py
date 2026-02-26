from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from google import genai
from google.genai import types
import base64
import requests
import threading
import time
from contextlib import asynccontextmanager

# --- BACKGROUND TASK TO PREVENT RENDER FROM SLEEPING ---
def keep_awake():
    """Background task to ping the server every 13 minutes to stop Render sleeping cycle."""
    url = "https://promptforge-vision.onrender.com/"
    while True:
        try:
            # Sleep for 13 minutes (780 seconds)
            time.sleep(780)
            print(f"Pinging {url} to keep server awake...")
            # We add a tiny timeout so our background thread doesn't hang
            requests.get(url, timeout=10)
            print("Ping successful!")
        except Exception as e:
            print(f"Self-ping failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the server starts
    print("Starting background keep-awake thread...")
    thread = threading.Thread(target=keep_awake, daemon=True)
    thread.start()
    yield # Server runs here
    # This runs when server closes
    print("Shutting down...")

# Create the FastAPI application
app = FastAPI(
    title="Gemini Image Gen API",
    description="API for enhancing prompts and generating images via Google Gemini 3 Pro.",
    lifespan=lifespan
)

# Define the structure of the JSON request body
class ImageRequest(BaseModel):
    title: str
    description: str
    add_text_overlay: bool = False
    enhance_prompt: bool = True        # Optional: skip prompt enhancement to save latency
    upload_imgbb: bool = True          # Optional: false to return raw base64 data-uri and save latency

# Define the structure of the JSON response body
class ImageResponse(BaseModel):
    image_url: str

@app.get("/")
def read_root():
    return {"message": "Gemini Image Generator API is running!"}

@app.post("/generate", response_model=ImageResponse)
def generate_image(
    request: ImageRequest,
    x_gemini_api_key: str = Header(..., description="Pass your Gemini API Key in this header"),
    x_imgbb_api_key: str = Header(..., description="Pass your ImgBB API Key in this header")
):
    """
    Accepts a title and description, enhances the prompt via Gemini 1.5 Flash, 
    and generates an image via Gemini 3 Pro Image Preview.
    """
    try:
        # Initialize Gemini client using the specific API key passed in the Header
        client = genai.Client(api_key=x_gemini_api_key)
        
        # === STEP 1: ENHANCE PROMPT WITH GEMINI TEXT MODEL (Optional) ===
        typography_instruction = ""
        if request.add_text_overlay:
            typography_instruction = f"\n\nCRITICAL: Explicitly instruct the image generator to integrate beautiful, professional typography directly into the visual composition (like a high-end Canva banner, magazine cover, or cinematic poster). The text must perfectly spell out '{request.title}' in a stunning font that matches the scene's aesthetic. Describe exactly where the text is placed and what it looks like."
            
        enhancement_instructions = f"""
        You are an expert AI Image Prompt Engineer. 
        Take the user's main subject and details and write a single highly detailed, 
        professional image generation prompt. Add deep descriptions of the lighting, 
        camera angle, mood, atmosphere, and artistic style. {typography_instruction}
        
        Make it sound like a cinematic masterpiece or brilliant conceptual artwork.
        Do NOT include introductory or concluding text, JUST output the enhanced image prompt.
        
        Main Subject: '{request.title}'
        Details & Setting: '{request.description}'
        """
        
        if request.enhance_prompt:
            # Use simple model for prompt engineering
            text_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=enhancement_instructions,
            )
            enhanced_prompt = text_response.text.strip()
        else:
            enhanced_prompt = f"{request.title}, {request.description}"
            if request.add_text_overlay:
                enhanced_prompt += f" Text overlay: '{request.title}'"
        
        # === STEP 2: GENERATE IMAGE WITH GEMINI IMAGE MODEL ===
        model = "gemini-3-pro-image-preview" 

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=enhanced_prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            image_config = types.ImageConfig(
                aspect_ratio="1:1",
                image_size="1K",
            ),
            response_modalities=["IMAGE"],
        )

        # Generate image natively (without streaming to avoid chunk processing overhead)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract binary data from the response
        image_data = None
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                     image_data = part.inline_data.data
                     break
                     
        if not image_data:
             raise HTTPException(status_code=500, detail="The model did not return image data.")
        
        # Convert binary image data to base64 string so it can be uploaded
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Optionally Upload to ImgBB (skip to save time)
        if request.upload_imgbb:
            # === STEP 3: UPLOAD TO IMGBB ===
            imgbb_response = requests.post(
                "https://api.imgbb.com/1/upload",
                data={
                    "key": x_imgbb_api_key,
                    "image": image_b64,
                }
            )
            
            imgbb_data = imgbb_response.json()
            if not imgbb_data.get("success"):
                error_msg = imgbb_data.get("error", {}).get("message", "Unknown Upload Error")
                raise HTTPException(status_code=500, detail=f"ImgBB Upload Failed: {error_msg}")
                
            image_url = imgbb_data["data"]["url"]
        else:
            # Directly return data URI, saving ImgBB uploading latency
            image_url = f"data:image/jpeg;base64,{image_b64}"
            
        return ImageResponse(
            image_url=image_url
        )

    except Exception as e:
        # Return elegant error if API key is invalid or Google returns an error
        raise HTTPException(status_code=500, detail=str(e))
