import os
import time
from pathlib import Path

import streamlit as st

from image_generator import generate_image, ASPECT_RATIOS
from prompt_enhancer import enhance_prompt

from gallery_manager import (
    load_history,
    add_to_history,
    delete_from_history,
    clear_history,
    delete_image_file,
)

from image_analyzer import (
    describe_image,
    ask_about_image,
    create_generation_prompt,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Image Generation Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_TITLE = "AI Image Generation Studio"
PROJECT_NAME = "DecodeLabs Project 3"
PROJECT_SUBTITLE = "Multimodal Image & Generation Studio"

GALLERY_DIR = Path("gallery")
UPLOAD_DIR = Path("uploads")

SUPPORTED_UPLOAD_TYPES = [
    "png",
    "jpg",
    "jpeg",
    "webp",
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

try:
    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

except Exception:
    pass


# ============================================================
# STYLE PRESETS
# ============================================================

STYLE_PRESETS = {

    "Photorealistic": (
        "photorealistic, ultra-detailed, realistic textures, "
        "natural lighting, professional photography"
    ),

    "Cinematic": (
        "cinematic composition, dramatic lighting, "
        "film-quality visuals, atmospheric depth"
    ),

    "Cyberpunk": (
        "cyberpunk aesthetic, neon lights, futuristic technology, "
        "high-tech environment, dark atmospheric mood"
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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 12px;
    }

    .status-card {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        text-align: center;
    }

    .small-text {
        font-size: 13px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {

    "generated_image": None,

    "enhanced_prompt": "",

    "generation_time": None,

    "image_description": "",

    "image_answer": "",

    "multimodal_prompt": "",

    "last_generation_prompt": "",

    "last_negative_prompt": "",

    "last_style": "Cinematic",

    "last_aspect_ratio": None,

    "last_num_steps": 4,

    "last_guidance_scale": 3.5,

    "last_seed": None,

    "last_uploaded_image": None,

}


for key, default_value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def show_user_error(
    message,
    technical_error=None
):
    """
    Display a friendly error message without exposing
    technical traceback information to normal users.
    """

    st.error(
        f"⚠️ {message}"
    )

    if technical_error:

        with st.expander(
            "🔍 Technical details"
        ):

            st.caption(
                f"{type(technical_error).__name__}: "
                f"{technical_error}"
            )


def reset_current_result():
    """
    Reset the current generated result.
    """

    st.session_state.generated_image = None

    st.session_state.enhanced_prompt = ""

    st.session_state.generation_time = None

    st.session_state.last_generation_prompt = ""

    st.session_state.last_negative_prompt = ""

    st.session_state.multimodal_prompt = ""


def validate_prompt(prompt):
    """
    Validate an image-generation prompt.
    """

    if prompt is None:
        return False

    if not prompt.strip():
        return False

    if len(prompt.strip()) < 3:
        return False

    return True


def validate_question(question):
    """
    Validate a multimodal question.
    """

    if question is None:
        return False

    if not question.strip():
        return False

    if len(question.strip()) < 2:
        return False

    return True


def save_uploaded_image(uploaded_file):
    """
    Safely save an uploaded image.
    """

    if uploaded_file is None:
        return None

    try:

        filename = Path(
            uploaded_file.name
        ).name

        extension = Path(
            filename
        ).suffix.lower()

        allowed_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        if extension not in allowed_extensions:

            raise ValueError(
                "Unsupported image format."
            )

        timestamp = int(
            time.time()
        )

        safe_filename = (
            f"upload_{timestamp}{extension}"
        )

        output_path = (
            UPLOAD_DIR / safe_filename
        )

        with open(
            output_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        return output_path

    except Exception as error:

        show_user_error(
            "The uploaded image could not be saved.",
            error
        )

        return None


def validate_generated_file(file_path):
    """
    Check whether a generated image actually exists.
    """

    if not file_path:
        return False

    path = Path(
        file_path
    )

    return (
        path.exists()
        and path.is_file()
    )


def clear_multimodal_results():
    """
    Clear previous multimodal analysis results.
    """

    st.session_state.image_description = ""

    st.session_state.image_answer = ""

    st.session_state.multimodal_prompt = ""


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎨 AI Image Generation Studio'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'DecodeLabs Project 3 • '
    'Multimodal Image & Generation Studio'
    '</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Feature status
# ------------------------------------------------------------

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:

    st.markdown(
        """
        <div class="status-card">
        🎨<br>
        <b>Image Generation</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col2:

    st.markdown(
        """
        <div class="status-card">
        ✨<br>
        <b>Prompt Enhancement</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col3:

    st.markdown(
        """
        <div class="status-card">
        👁️<br>
        <b>Multimodal AI</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col4:

    st.markdown(
        """
        <div class="status-card">
        🖼️<br>
        <b>Gallery</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Generation Settings"
    )

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    style = st.selectbox(
        "🎨 Visual Style",
        list(
            STYLE_PRESETS.keys()
        ),
        index=1,
    )

    st.caption(
        STYLE_PRESETS[style]
    )

    # --------------------------------------------------------
    # ASPECT RATIO
    # --------------------------------------------------------

    aspect_ratio = st.selectbox(
        "📐 Aspect Ratio",
        list(
            ASPECT_RATIOS.keys()
        ),
        index=2,
    )

    width, height = ASPECT_RATIOS[
        aspect_ratio
    ]

    st.info(
        f"Resolution: **{width} × {height} px**"
    )

    # --------------------------------------------------------
    # INFERENCE STEPS
    # --------------------------------------------------------

    num_steps = st.slider(
        "⚙️ Inference Steps",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
    )

    # --------------------------------------------------------
    # GUIDANCE
    # --------------------------------------------------------

    guidance_scale = st.slider(
        "🎯 Guidance Scale",
        min_value=1.0,
        max_value=10.0,
        value=3.5,
        step=0.5,
    )

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    use_seed = st.checkbox(
        "🎲 Use fixed seed"
    )

    seed = None

    if use_seed:

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999999,
            value=42,
            step=1,
        )

    st.divider()

    # --------------------------------------------------------
    # CURRENT SETTINGS
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Current Settings"
    )

    st.caption(
        f"Style: **{style}**"
    )

    st.caption(
        f"Ratio: **{aspect_ratio}**"
    )

    st.caption(
        f"Steps: **{num_steps}**"
    )

    st.caption(
        f"Guidance: **{guidance_scale}**"
    )

    # --------------------------------------------------------
    # CLEAR CURRENT RESULT
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Current Result",
        use_container_width=True,
    ):

        reset_current_result()

        st.success(
            "Current result cleared."
        )

        st.rerun()


# ============================================================
# MAIN TABS
# ============================================================

generate_tab, multimodal_tab, gallery_tab = st.tabs(
    [
        "✨ Generate",
        "👁️ Multimodal Studio",
        "🖼️ Gallery",
    ]
)


# ============================================================
# TAB 1 — GENERATE
# ============================================================

with generate_tab:

    left_column, right_column = st.columns(
        [1, 1.35],
        gap="large",
    )

    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left_column:

        st.markdown(
            "### 📝 Create Your Image"
        )

        prompt = st.text_area(
            "Describe the image you want",
            placeholder=(
                "Example:\n"
                "A futuristic Indian city at sunset "
                "with glowing skyscrapers and flying vehicles..."
            ),
            height=160,
            key="main_prompt",
        )

        # ----------------------------------------------------
        # PROMPT ENHANCEMENT
        # ----------------------------------------------------

        if st.button(
            "✨ Enhance Prompt with AI",
            use_container_width=True,
            key="enhance_prompt_button",
        ):

            if not validate_prompt(prompt):

                st.warning(
                    "⚠️ Please enter a meaningful image "
                    "description first."
                )

            else:

                with st.spinner(
                    "✨ Gemini is enhancing your prompt..."
                ):

                    try:

                        enhanced = enhance_prompt(
                            user_prompt=prompt,
                            style=style,
                        )

                        if not enhanced:

                            raise ValueError(
                                "Gemini returned an empty prompt."
                            )

                        st.session_state.enhanced_prompt = (
                            enhanced
                        )

                        st.success(
                            "✅ Prompt enhanced successfully!"
                        )

                    except Exception as error:

                        show_user_error(
                            "We couldn't enhance your prompt. "
                            "Please check your Gemini configuration "
                            "and try again.",
                            error
                        )

        # ----------------------------------------------------
        # ENHANCED PROMPT
        # ----------------------------------------------------

        if st.session_state.enhanced_prompt:

            st.markdown(
                "### ✨ AI Enhanced Prompt"
            )

            st.text_area(
                "Enhanced prompt",
                value=(
                    st.session_state.enhanced_prompt
                ),
                height=220,
                disabled=True,
                key="enhanced_prompt_display",
            )

        # ----------------------------------------------------
        # NEGATIVE PROMPT
        # ----------------------------------------------------

        negative_prompt = st.text_area(
            "🚫 Negative Prompt",
            placeholder=(
                "blurry, low quality, distorted, "
                "watermark, text, duplicate objects"
            ),
            height=100,
            key="negative_prompt",
        )

        st.divider()

        # ----------------------------------------------------
        # GENERATE BUTTON
        # ----------------------------------------------------

        generate_button = st.button(
            "🚀 Generate Image",
            type="primary",
            use_container_width=True,
            key="generate_image_button",
        )

    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right_column:

        st.markdown(
            "### 🖼️ Generated Artwork"
        )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        if generate_button:

            if not validate_prompt(prompt):

                st.warning(
                    "⚠️ Please enter an image description "
                    "before generating."
                )

            else:

                # --------------------------------------------
                # SELECT PROMPT
                # --------------------------------------------

                if st.session_state.enhanced_prompt:

                    generation_prompt = (
                        st.session_state.enhanced_prompt
                    )

                else:

                    generation_prompt = (
                        f"{prompt.strip()}, "
                        f"{STYLE_PRESETS[style]}"
                    )

                # --------------------------------------------
                # SAVE SETTINGS
                # --------------------------------------------

                st.session_state.last_generation_prompt = (
                    generation_prompt
                )

                st.session_state.last_negative_prompt = (
                    negative_prompt
                )

                st.session_state.last_style = style

                st.session_state.last_aspect_ratio = (
                    aspect_ratio
                )

                st.session_state.last_num_steps = (
                    num_steps
                )

                st.session_state.last_guidance_scale = (
                    guidance_scale
                )

                st.session_state.last_seed = seed

                # --------------------------------------------
                # GENERATE
                # --------------------------------------------

                with st.spinner(
                    "🎨 Creating your artwork..."
                ):

                    start_time = time.time()

                    try:

                        timestamp = int(
                            time.time()
                        )

                        output_path = (
                            GALLERY_DIR
                            / f"image_{timestamp}.png"
                        )

                        generated_file = generate_image(
                            prompt=generation_prompt,
                            output_path=str(
                                output_path
                            ),
                            aspect_ratio=aspect_ratio,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt,
                            seed=seed,
                        )

                        elapsed_time = (
                            time.time()
                            - start_time
                        )

                        # ------------------------------------
                        # VALIDATE OUTPUT
                        # ------------------------------------

                        if not validate_generated_file(
                            generated_file
                        ):

                            raise FileNotFoundError(
                                "The image generator completed "
                                "but the output image was not found."
                            )

                        # ------------------------------------
                        # SAVE GALLERY
                        # ------------------------------------

                        try:

                            add_to_history(
                                image_path=generated_file,
                                prompt=generation_prompt,
                                style=style,
                                aspect_ratio=aspect_ratio,
                                width=width,
                                height=height,
                                generation_time=elapsed_time,
                                seed=seed,
                            )

                        except Exception as gallery_error:

                            show_user_error(
                                "The image was generated, "
                                "but it could not be added to "
                                "the gallery.",
                                gallery_error
                            )

                        # ------------------------------------
                        # SESSION STATE
                        # ------------------------------------

                        st.session_state.generated_image = (
                            str(generated_file)
                        )

                        st.session_state.generation_time = (
                            elapsed_time
                        )

                        st.success(
                            "🎉 Image generated successfully!"
                        )

                    except Exception as error:

                        show_user_error(
                            "Image generation failed. "
                            "Please check your prompt and "
                            "generation settings, then try again.",
                            error
                        )

        # ----------------------------------------------------
        # DISPLAY GENERATED IMAGE
        # ----------------------------------------------------

        if st.session_state.generated_image:

            image_path = Path(
                st.session_state.generated_image
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption=(
                        f"{style} • {aspect_ratio}"
                    ),
                    use_container_width=True,
                )

                # --------------------------------------------
                # GENERATION TIME
                # --------------------------------------------

                if st.session_state.generation_time:

                    st.success(
                        "⏱️ Generation time: "
                        f"{st.session_state.generation_time:.2f} seconds"
                    )

                # --------------------------------------------
                # DOWNLOAD
                # --------------------------------------------

                try:

                    with open(
                        image_path,
                        "rb",
                    ) as image_file:

                        st.download_button(
                            "⬇️ Download Image",
                            data=image_file,
                            file_name=(
                                "decodeLabs_artwork.png"
                            ),
                            mime="image/png",
                            use_container_width=True,
                        )

                except Exception as error:

                    show_user_error(
                        "The image could not be prepared "
                        "for download.",
                        error
                    )

                # --------------------------------------------
                # REGENERATE
                # --------------------------------------------

                if st.button(
                    "🔄 Regenerate Image",
                    use_container_width=True,
                    key="regenerate_button",
                ):

                    with st.spinner(
                        "🔄 Regenerating your artwork..."
                    ):

                        try:

                            regenerate_prompt = (
                                st.session_state.last_generation_prompt
                            )

                            regenerate_path = (
                                GALLERY_DIR
                                / f"image_{int(time.time())}.png"
                            )

                            regenerate_start = (
                                time.time()
                            )

                            regenerated_file = (
                                generate_image(
                                    prompt=regenerate_prompt,
                                    output_path=str(
                                        regenerate_path
                                    ),
                                    aspect_ratio=(
                                        st.session_state.last_aspect_ratio
                                    ),
                                    num_inference_steps=(
                                        st.session_state.last_num_steps
                                    ),
                                    guidance_scale=(
                                        st.session_state.last_guidance_scale
                                    ),
                                    negative_prompt=(
                                        st.session_state.last_negative_prompt
                                    ),
                                    seed=(
                                        st.session_state.last_seed
                                    ),
                                )
                            )

                            regenerate_elapsed = (
                                time.time()
                                - regenerate_start
                            )

                            if not validate_generated_file(
                                regenerated_file
                            ):

                                raise FileNotFoundError(
                                    "Regenerated image was not found."
                                )

                            add_to_history(
                                image_path=regenerated_file,
                                prompt=regenerate_prompt,
                                style=(
                                    st.session_state.last_style
                                ),
                                aspect_ratio=(
                                    st.session_state.last_aspect_ratio
                                ),
                                width=width,
                                height=height,
                                generation_time=(
                                    regenerate_elapsed
                                ),
                                seed=(
                                    st.session_state.last_seed
                                ),
                            )

                            st.session_state.generated_image = (
                                str(regenerated_file)
                            )

                            st.session_state.generation_time = (
                                regenerate_elapsed
                            )

                            st.success(
                                "🔄 Image regenerated successfully!"
                            )

                            st.rerun()

                        except Exception as error:

                            show_user_error(
                                "Regeneration failed. "
                                "Please try again.",
                                error
                            )

            else:

                st.warning(
                    "⚠️ The generated image file "
                    "could not be found."
                )

        else:

            st.info(
                "🎨 Your generated artwork will appear here."
            )


# ============================================================
# TAB 2 — MULTIMODAL STUDIO
# ============================================================

with multimodal_tab:

    st.markdown(
        "## 👁️ Multimodal Image Studio"
    )

    st.write(
        "Upload an image and use Gemini AI to "
        "understand, analyze, and transform it "
        "into a new generation prompt."
    )

    st.divider()

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=SUPPORTED_UPLOAD_TYPES,
        key="multimodal_upload",
    )

    if uploaded_file:

        # --------------------------------------------
        # DISPLAY IMAGE
        # --------------------------------------------

        try:

            st.image(
                uploaded_file,
                caption="Uploaded Image",
                use_container_width=True,
            )

        except Exception as error:

            show_user_error(
                "The uploaded image could not be displayed.",
                error
            )

        # --------------------------------------------
        # SAVE IMAGE
        # --------------------------------------------

        current_upload_name = (
            uploaded_file.name
        )

        if (
            st.session_state.last_uploaded_image
            != current_upload_name
        ):

            clear_multimodal_results()

            st.session_state.last_uploaded_image = (
                current_upload_name
            )

        temp_path = save_uploaded_image(
            uploaded_file
        )

        if temp_path:

            st.success(
                "✅ Image uploaded successfully."
            )

            st.divider()

            # ====================================================
            # IMAGE DESCRIPTION
            # ====================================================

            st.markdown(
                "### 📝 AI Image Description"
            )

            st.write(
                "Gemini will describe the visual contents "
                "of the uploaded image."
            )

            if st.button(
                "📝 Describe Image",
                use_container_width=True,
                key="describe_image_button",
            ):

                with st.spinner(
                    "👁️ Gemini is analyzing the image..."
                ):

                    try:

                        description = describe_image(
                            temp_path
                        )

                        if not description:

                            raise ValueError(
                                "Gemini returned an empty description."
                            )

                        st.session_state.image_description = (
                            description
                        )

                        st.success(
                            "✅ Image analysis completed!"
                        )

                    except Exception as error:

                        show_user_error(
                            "We couldn't analyze this image. "
                            "Please try another image.",
                            error
                        )

            if st.session_state.image_description:

                st.info(
                    st.session_state.image_description
                )

            st.divider()

            # ====================================================
            # ASK GEMINI
            # ====================================================

            st.markdown(
                "### ❓ Ask Gemini About This Image"
            )

            question = st.text_input(
                "Ask a question about the uploaded image",
                placeholder=(
                    "Example: What are the main objects "
                    "in this image?"
                ),
                key="image_question",
            )

            if st.button(
                "🤖 Ask Gemini",
                use_container_width=True,
                key="ask_image_button",
            ):

                if not validate_question(
                    question
                ):

                    st.warning(
                        "⚠️ Please enter a question "
                        "about the image."
                    )

                else:

                    with st.spinner(
                        "🤖 Gemini is analyzing your question..."
                    ):

                        try:

                            answer = ask_about_image(
                                temp_path,
                                question,
                            )

                            if not answer:

                                raise ValueError(
                                    "Gemini returned an empty answer."
                                )

                            st.session_state.image_answer = (
                                answer
                            )

                            st.success(
                                "✅ Question answered!"
                            )

                        except Exception as error:

                            show_user_error(
                                "We couldn't answer that question. "
                                "Please try asking it differently.",
                                error
                            )

            if st.session_state.image_answer:

                st.markdown(
                    "#### 🤖 Gemini Answer"
                )

                st.success(
                    st.session_state.image_answer
                )

            st.divider()

            # ====================================================
            # IMAGE → GENERATION PROMPT
            # ====================================================

            st.markdown(
                "### ✨ Image → Generation Prompt"
            )

            st.write(
                "Gemini will analyze the uploaded image "
                "and create a detailed text-to-image prompt."
            )

            if st.button(
                "✨ Create Generation Prompt",
                use_container_width=True,
                key="create_generation_prompt_button",
            ):

                with st.spinner(
                    "✨ Creating generation prompt..."
                ):

                    try:

                        generated_prompt = (
                            create_generation_prompt(
                                temp_path,
                                style=style,
                            )
                        )

                        if not generated_prompt:

                            raise ValueError(
                                "Gemini returned an empty generation prompt."
                            )

                        st.session_state.multimodal_prompt = (
                            generated_prompt
                        )

                        st.session_state.enhanced_prompt = (
                            generated_prompt
                        )

                        st.success(
                            "🎉 Generation prompt created successfully!"
                        )

                    except Exception as error:

                        show_user_error(
                            "We couldn't create a generation prompt "
                            "from this image. Please try again.",
                            error
                        )

            if st.session_state.multimodal_prompt:

                st.markdown(
                    "#### ✨ AI Generated Prompt"
                )

                st.text_area(
                    "Generated prompt",
                    value=(
                        st.session_state.multimodal_prompt
                    ),
                    height=220,
                    disabled=True,
                    key="multimodal_prompt_display",
                )

                st.info(
                    "💡 Go to the ✨ Generate tab and click "
                    "🚀 Generate Image to create new artwork "
                    "from this prompt."
                )

        else:

            st.warning(
                "⚠️ The image could not be uploaded. "
                "Please try another file."
            )

    else:

        st.info(
            "📤 Upload an image to begin multimodal analysis."
        )


# ============================================================
# TAB 3 — GALLERY
# ============================================================

with gallery_tab:

    st.markdown(
        "## 🖼️ Generation Gallery"
    )

    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------

    try:

        history = load_history()

    except Exception as error:

        history = []

        show_user_error(
            "The gallery history could not be loaded.",
            error
        )

    # --------------------------------------------------------
    # EMPTY GALLERY
    # --------------------------------------------------------

    if not history:

        st.info(
            "🖼️ No generated images yet. "
            "Create your first artwork!"
        )

    else:

        st.write(
            f"**{len(history)} generated image(s)**"
        )

        st.divider()

        # ----------------------------------------------------
        # GALLERY GRID
        # ----------------------------------------------------

        for index in range(
            0,
            len(history),
            3,
        ):

            row = history[
                index:index + 3
            ]

            columns = st.columns(
                len(row)
            )

            for column, item in zip(
                columns,
                row,
            ):

                with column:

                    try:

                        image_path = Path(
                            item["image_path"]
                        )

                        if image_path.exists():

                            st.image(
                                str(image_path),
                                use_container_width=True,
                            )

                            st.caption(
                                f"🎨 Style: "
                                f"{item.get('style', 'Unknown')}"
                            )

                            st.caption(
                                f"📐 Ratio: "
                                f"{item.get('aspect_ratio', 'Unknown')}"
                            )

                            st.caption(
                                f"🖼️ Resolution: "
                                f"{item.get('resolution', 'Unknown')}"
                            )

                            st.caption(
                                f"⏱️ Time: "
                                f"{item.get('generation_time', 'Unknown')} sec"
                            )

                            # --------------------------------
                            # DOWNLOAD
                            # --------------------------------

                            try:

                                with open(
                                    image_path,
                                    "rb",
                                ) as image_file:

                                    st.download_button(
                                        "⬇️ Download",
                                        data=image_file,
                                        file_name=(
                                            f"artwork_"
                                            f"{item.get('id', index)}.png"
                                        ),
                                        mime="image/png",
                                        key=(
                                            f"download_"
                                            f"{item.get('id', index)}"
                                        ),
                                        use_container_width=True,
                                    )

                            except Exception as error:

                                show_user_error(
                                    "This image could not "
                                    "be downloaded.",
                                    error
                                )

                            # --------------------------------
                            # VIEW PROMPT
                            # --------------------------------

                            with st.expander(
                                "📝 View Prompt"
                            ):

                                st.write(
                                    item.get(
                                        "prompt",
                                        "Prompt unavailable."
                                    )
                                )

                            # --------------------------------
                            # DELETE
                            # --------------------------------

                            item_id = item.get(
                                "id"
                            )

                            if st.button(
                                "🗑️ Delete",
                                key=f"delete_{item_id}",
                                use_container_width=True,
                            ):

                                try:

                                    delete_image_file(
                                        item["image_path"]
                                    )

                                    delete_from_history(
                                        item_id
                                    )

                                    st.success(
                                        "Image deleted."
                                    )

                                    st.rerun()

                                except Exception as error:

                                    show_user_error(
                                        "The image could not "
                                        "be deleted.",
                                        error
                                    )

                        else:

                            st.warning(
                                "⚠️ Image file not found."
                            )

                    except Exception as error:

                        show_user_error(
                            "This gallery item could not "
                            "be displayed.",
                            error
                        )

        st.divider()

        # ----------------------------------------------------
        # CLEAR ENTIRE GALLERY
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Entire Gallery",
            use_container_width=True,
        ):

            try:

                for item in history:

                    image_path = item.get(
                        "image_path"
                    )

                    if image_path:

                        delete_image_file(
                            image_path
                        )

                clear_history()

                reset_current_result()

                st.success(
                    "✅ Gallery cleared successfully."
                )

                st.rerun()

            except Exception as error:

                show_user_error(
                    "The gallery could not be cleared.",
                    error
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; opacity:0.7;">

    <b>🎨 DecodeLabs Project 3</b><br>

    Multimodal Image & Generation Studio<br>

    Stage 7 • Error Handling + UI Polish

    </div>
    """,
    unsafe_allow_html=True,
)