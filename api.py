from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from google import genai
from google.genai import types
import base64

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
    enhanced_prompt: str
    image_base64: str

@app.get("/")
def read_root():
    return {"message": "Gemini Image Generator API is running!"}

@app.post("/generate", response_model=ImageResponse)
def generate_image(
    request: ImageRequest,
    x_gemini_api_key: str = Header(..., description="Pass your Gemini API Key in this header")
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
        
        # Convert binary image data to base64 string so it can be returned via JSON
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        return ImageResponse(
            enhanced_prompt=enhanced_prompt,
            image_base64=image_b64
        )

    except Exception as e:
        # Return elegant error if API key is invalid or Google returns an error
        raise HTTPException(status_code=500, detail=str(e))
