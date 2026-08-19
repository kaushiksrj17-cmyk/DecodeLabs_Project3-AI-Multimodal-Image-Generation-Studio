import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is missing.\n"
        "Please add your Hugging Face token to the .env file."
    )


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# ============================================================
# ASPECT RATIO PRESETS
# ============================================================

ASPECT_RATIOS = {
    "Square (1:1)": (1024, 1024),
    "Portrait (3:4)": (768, 1024),
    "Landscape (4:3)": (1024, 768),
    "Wide (16:9)": (1024, 576),
    "Tall (9:16)": (576, 1024),
}


# ============================================================
# IMAGE GENERATION FUNCTION
# ============================================================

def generate_image(
    prompt: str,
    output_path: str = "generated_image.png",
    aspect_ratio: str = "Square (1:1)",
    num_inference_steps: int = 4,
    guidance_scale: float = 3.5,
    negative_prompt: str = "",
    seed: int | None = None,
):
    """
    Generate an image using Hugging Face Inference Providers.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            f"Unsupported aspect ratio: {aspect_ratio}"
        )

    width, height = ASPECT_RATIOS[aspect_ratio]

    parameters = {
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    if negative_prompt and negative_prompt.strip():
        parameters["negative_prompt"] = negative_prompt.strip()

    if seed is not None:
        parameters["seed"] = seed

    image = client.text_to_image(
        prompt=prompt.strip(),
        model="black-forest-labs/FLUX.1-schnell",
        **parameters
    )

    output_file = Path(output_path)

    image.save(output_file)

    return output_file