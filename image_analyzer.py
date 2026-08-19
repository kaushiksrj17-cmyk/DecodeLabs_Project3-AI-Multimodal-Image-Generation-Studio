import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add GEMINI_API_KEY to your .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# MODELS
# ============================================================

PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3.5-flash"


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# ============================================================
# MIME TYPE HELPER
# ============================================================

def _get_mime_type(image_path):
    """
    Return the MIME type for an image file.
    """

    image_path = Path(image_path)

    suffix = image_path.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats: "
            f"{', '.join(SUPPORTED_IMAGE_TYPES.keys())}"
        )

    return SUPPORTED_IMAGE_TYPES[suffix]


# ============================================================
# IMAGE VALIDATION
# ============================================================

def _validate_image(image_path):
    """
    Validate that the image exists and has a supported format.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Provided path is not a file: {image_path}"
        )

    _get_mime_type(image_path)

    return image_path


# ============================================================
# GEMINI IMAGE REQUEST
# ============================================================

def _analyze_image(
    image_path,
    prompt
):
    """
    Send an image and prompt to Gemini.

    Uses the primary model first and the fallback model
    if the primary model fails.
    """

    image_path = _validate_image(image_path)

    mime_type = _get_mime_type(
        image_path
    )

    image_bytes = image_path.read_bytes()

    contents = [
        prompt,
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )
    ]

    # --------------------------------------------------------
    # PRIMARY MODEL
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=contents
        )

        if response and response.text:

            return response.text.strip()

    except Exception as primary_error:

        primary_exception = primary_error

    # --------------------------------------------------------
    # FALLBACK MODEL
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=contents
        )

        if response and response.text:

            return response.text.strip()

    except Exception as fallback_error:

        raise RuntimeError(
            "Gemini image analysis failed.\n\n"
            f"Primary model error: {primary_exception}\n\n"
            f"Fallback model error: {fallback_error}"
        )

    raise RuntimeError(
        "Gemini returned an empty response."
    )


# ============================================================
# IMAGE DESCRIPTION
# ============================================================

def describe_image(image_path):
    """
    Analyze an image and generate a detailed description.
    """

    prompt = """
Analyze this image carefully and provide a detailed
description.

Include:

1. Main subjects
2. People and objects
3. Environment
4. Background
5. Colors
6. Lighting
7. Composition
8. Camera perspective
9. Visible actions
10. Important visual details
11. Overall atmosphere or mood

Be accurate and describe only what can reasonably be
observed in the image.

Return the description in clear, natural language.
"""

    return _analyze_image(
        image_path,
        prompt
    )


# ============================================================
# ASK A QUESTION ABOUT AN IMAGE
# ============================================================

def ask_about_image(
    image_path,
    question
):
    """
    Answer a user's question about an uploaded image.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    prompt = f"""
Analyze the provided image and answer the user's question.

User question:
{question.strip()}

Instructions:

- Base the answer on the visual information in the image.
- Be accurate and concise.
- If the information cannot be determined from the image,
  clearly say that it cannot be determined.
- Do not invent details.

Return only the answer.
"""

    return _analyze_image(
        image_path,
        prompt
    )


# ============================================================
# IMAGE → GENERATION PROMPT
# ============================================================

def create_generation_prompt(
    image_path,
    style="Cinematic"
):
    """
    Analyze an image and convert its visual characteristics
    into a detailed text-to-image generation prompt.
    """

    if not style:
        style = "Cinematic"

    prompt = f"""
Analyze the provided image carefully and create a
high-quality prompt for an AI image generation model.

Requested visual style:
{style}

Study the image and describe its important visual
characteristics, including:

- Main subject
- People
- Objects
- Environment
- Background
- Composition
- Camera angle
- Perspective
- Framing
- Lighting
- Shadows
- Colors
- Materials
- Textures
- Clothing
- Facial expressions, when visible
- Pose and body positioning, when visible
- Depth
- Atmosphere
- Mood
- Important visual details

Transform these observations into a polished,
detailed text-to-image prompt.

Preserve the important visual characteristics of the
original image while adapting the description to the
requested style.

Do not mention that you are analyzing an uploaded image.

Return ONLY the final image-generation prompt.

Do not include:
- explanations
- headings
- bullet points
- quotation marks
- commentary
"""

    return _analyze_image(
        image_path,
        prompt
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MULTIMODAL IMAGE ANALYZER TEST")
    print("=" * 60)

    sample_image = Path(
        "assets/sample.jpg"
    )

    print()
    print(f"Image: {sample_image}")
    print(f"Primary model: {PRIMARY_MODEL}")
    print(f"Fallback model: {FALLBACK_MODEL}")

    # --------------------------------------------------------
    # TEST IMAGE DESCRIPTION
    # --------------------------------------------------------

    print()
    print("Testing image description...")
    print()

    try:

        description = describe_image(
            sample_image
        )

        print("-" * 60)
        print("IMAGE DESCRIPTION")
        print("-" * 60)
        print(description)

    except Exception as error:

        print()
        print("=" * 60)
        print("IMAGE ANALYSIS FAILED")
        print("=" * 60)
        print()
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")