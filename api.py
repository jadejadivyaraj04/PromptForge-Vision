from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from google import genai
from google.genai import types
import base64
import requests
import io
from PIL import Image, ImageDraw, ImageFont

# Create the FastAPI application
app = FastAPI(
    title="Gemini Image Gen API",
    description="API for enhancing prompts and generating images via Google Gemini 3 Pro."
)

# Define the structure of the JSON request body
class ImageRequest(BaseModel):
    title: str
    description: str

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
        
        # === STEP 1: ENHANCE PROMPT WITH GEMINI TEXT MODEL ===
        enhancement_instructions = f"""
        You are an expert AI Image Prompt Engineer. 
        Take the user's main subject and details and write a single highly detailed, 
        professional image generation prompt. Add deep descriptions of the lighting, 
        camera angle, mood, atmosphere, and artistic style. 
        
        Make it sound like a cinematic masterpiece or brilliant conceptual artwork.
        Do NOT include introductory or concluding text, JUST output the enhanced image prompt.
        
        Main Subject: '{request.title}'
        Details & Setting: '{request.description}'
        """
        
        # Use simple model for prompt engineering
        text_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=enhancement_instructions,
        )
        enhanced_prompt = text_response.text.strip()
        
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

        response_stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract binary data from the stream response
        image_data = None
        for chunk in response_stream:
            if chunk.parts is None:
                continue
            for part in chunk.parts:
                if part.inline_data and part.inline_data.data:
                    image_data = part.inline_data.data
                    break
                    
        if not image_data:
             raise HTTPException(status_code=500, detail="The model did not return an image data.")
        
        # === NEW STEP: STAMP TEXT OVER IMAGE ===
        try:
            # 1. Load image and prepare for drawing a transparent overlay
            img = Image.open(io.BytesIO(image_data)).convert("RGBA")
            width, height = img.size
            
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # 2. Draw a dark semi-transparent banner at the bottom 20%
            banner_height = int(height * 0.2)
            banner_top = height - banner_height
            draw.rectangle(
                [(0, banner_top), (width, height)],
                fill=(0, 0, 0, 200) # Dark gradient/banner
            )
            
            # Combine the overlay
            img = Image.alpha_composite(img, overlay)
            draw_text = ImageDraw.Draw(img)
            
            # 3. Load Fonts (Dynamic sizing based on image resolution)
            try:
                title_font = ImageFont.load_default(size=int(height * 0.05))
                desc_font = ImageFont.load_default(size=int(height * 0.035))
            except TypeError:
                title_font = ImageFont.load_default()
                desc_font = title_font
                
            # 4. Draw the text inside the banner
            pad_x = int(width * 0.04)
            pad_y = int(height * 0.03)
            
            # Draw Title (White)
            draw_text.text((pad_x, banner_top + pad_y), request.title, font=title_font, fill=(255, 255, 255, 255))
            
            # Draw Description (Light Gray) just beneath the title
            draw_text.text((pad_x, banner_top + pad_y + int(height * 0.07)), request.description[:100] + ("..." if len(request.description) > 100 else ""), font=desc_font, fill=(200, 200, 200, 255))
            
            # 5. Extract finalized image byte data
            final_image_data = io.BytesIO()
            img.convert("RGB").save(final_image_data, format="PNG")
            image_data = final_image_data.getvalue()
        except Exception as e:
            print(f"Non-fatal error during text stamping: {e}")
            pass # Fails gracefully, proceeds with unstamped image
        
        # Convert binary image data to base64 string so it can be uploaded
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
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
        
        return ImageResponse(
            image_url=image_url
        )

    except Exception as e:
        # Return elegant error if API key is invalid or Google returns an error
        raise HTTPException(status_code=500, detail=str(e))
