import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add your Gemini API key to the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# PROMPT ENHANCER
# ============================================================

def enhance_prompt(
    user_prompt: str,
    style: str = "Cinematic"
) -> str:
    """
    Enhance a user's basic image-generation prompt
    using Gemini AI.
    """

    if not user_prompt or not user_prompt.strip():
        raise ValueError(
            "Image prompt cannot be empty."
        )

    style_description = {
        "Photorealistic": (
            "photorealistic, realistic textures, "
            "natural lighting, professional photography"
        ),
        "Cinematic": (
            "cinematic composition, dramatic lighting, "
            "film-quality visuals, atmospheric depth"
        ),
        "Cyberpunk": (
            "cyberpunk aesthetic, neon lights, "
            "futuristic technology, high-tech environment"
        ),
        "Anime": (
            "high-quality anime artwork, expressive details, "
            "vibrant colors, polished anime illustration"
        ),
        "3D Render": (
            "high-quality 3D render, realistic materials, "
            "detailed geometry, professional studio lighting"
        ),
        "Watercolor": (
            "beautiful watercolor painting, soft brush strokes, "
            "artistic textures, traditional watercolor aesthetic"
        ),
        "Fantasy": (
            "epic fantasy artwork, magical atmosphere, "
            "dramatic environment, intricate details"
        ),
        "Minimalist": (
            "minimalist composition, clean design, "
            "simple shapes, balanced negative space"
        ),
    }

    selected_style = style_description.get(
        style,
        style
    )

    instruction = f"""
You are an expert AI image prompt engineer.

Transform the user's simple image description into a
high-quality, detailed prompt suitable for an AI image
generation model.

User description:
{user_prompt.strip()}

Visual style:
{selected_style}

Create a detailed prompt that includes, when appropriate:

- Main subject
- Environment
- Composition
- Camera perspective
- Lighting
- Colors
- Materials and textures
- Atmosphere
- Depth
- Important visual details
- Artistic style

Keep the original meaning of the user's request.
Do not add unrelated objects or concepts.

Return ONLY the final enhanced image-generation prompt.
Do not include explanations, headings, quotation marks,
or labels.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=instruction,
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty prompt."
        )

    return response.text.strip()