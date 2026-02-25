import streamlit as st
import os
from google import genai
from google.genai import types

# Provide a sidebar input for the API Key so it is not hardcoded
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", help="Get this from Google AI Studio")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

st.set_page_config(page_title="Gemini Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 Gemini Image Generator Dashboard")
st.markdown("Easily generate images using the **gemini-3-pro-image-preview** model.")

# Create inputs for Title and Description
col1, col2 = st.columns([1, 2])
with col1:
    image_title = st.text_input("Main Subject (Title):", placeholder="e.g. A flying car")
with col2:
    image_description = st.text_area("Details & Settings (Description):", placeholder="e.g. Night time, neon cyberpunk city, rain reflecting on metal", height=100)

if st.button("✨ Generate Image", type="primary", use_container_width=True):
    if not image_title.strip() or not image_description.strip():
        st.warning("⚠️ Please provide both a Main Subject and Details.")
    else:
        with st.spinner("⏳ Compiling your prompt & creating magic (this may take a few seconds)..."):
            try:
                # Initialize the Gemini Client
                client = genai.Client()
                
                # --- STEP 1: ENHANCE THE PROMPT FOR FREE ---
                st.info("🤖 AI is enhancing your prompt and adding details...")
                
                enhancement_instructions = f"""
                You are an expert AI Image Prompt Engineer. 
                Take the user's main subject and details and write a single highly detailed, 
                professional image generation prompt. Add deep descriptions of the lighting, 
                camera angle, mood, atmosphere, and artistic style. 
                
                Make it sound like a cinematic masterpiece or brilliant conceptual artwork.
                Do NOT include introductory or concluding text, JUST output the enhanced image prompt.
                
                Main Subject: '{image_title}'
                Details & Setting: '{image_description}'
                """
                
                # We use the free-tier, hyper-fast text model for this
                text_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=enhancement_instructions,
                )
                
                enhanced_prompt = text_response.text.strip()
                st.success(f"**✨ Enhanced Prompt generated:**\n\n_{enhanced_prompt}_")
                
                # --- STEP 2: GENERATE THE IMAGE ---
                st.info("🎨 Now painting the image...")
                
                model = "gemini-3-pro-image-preview" 

                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=enhanced_prompt),
                        ],
                    ),
                ]
                
                # Generating Image config from user code
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
                
                image_found = False
                for chunk in response_stream:
                    if chunk.parts is None:
                        continue
                        
                    for part in chunk.parts:
                        if part.inline_data and part.inline_data.data:
                            image_found = True
                            st.success("✅ Image generated successfully!")
                            st.image(part.inline_data.data, caption=f"Prompt: {image_title}", use_container_width=True)
                            
                            # Provide a download button
                            st.download_button(
                                label="⬇️ Download Image",
                                data=part.inline_data.data,
                                file_name="generated_image.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        elif part.text:
                            # It might generate a text response along with or instead of an image
                            if part.text.strip():
                                st.info(f"Model Message: {part.text}")

                if not image_found:
                    st.info("The model completed but didn't return an image. (It may have been blocked by safety filters or a typo in the prompt).")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
