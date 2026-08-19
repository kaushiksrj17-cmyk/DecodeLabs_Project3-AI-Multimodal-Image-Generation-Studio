import json
from pathlib import Path
from datetime import datetime


# ============================================================
# GALLERY CONFIGURATION
# ============================================================

GALLERY_DIR = Path("gallery")
HISTORY_FILE = GALLERY_DIR / "history.json"


# ============================================================
# INITIALIZE GALLERY
# ============================================================

def initialize_gallery():

    GALLERY_DIR.mkdir(
        exist_ok=True
    )

    if not HISTORY_FILE.exists():

        HISTORY_FILE.write_text(
            "[]",
            encoding="utf-8"
        )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    initialize_gallery()

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            history = json.load(file)

        return history

    except (json.JSONDecodeError, OSError):

        return []


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history):

    initialize_gallery()

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# ADD IMAGE TO HISTORY
# ============================================================

def add_to_history(
    image_path,
    prompt,
    style,
    aspect_ratio,
    width,
    height,
    generation_time,
    seed=None
):

    history = load_history()

    item = {
        "id": datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        ),

        "image_path": str(
            image_path
        ),

        "prompt": prompt,

        "style": style,

        "aspect_ratio": aspect_ratio,

        "resolution": f"{width} x {height}",

        "generation_time": round(
            generation_time,
            2
        ),

        "seed": seed,

        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    history.insert(
        0,
        item
    )

    save_history(
        history
    )

    return item


# ============================================================
# DELETE HISTORY ITEM
# ============================================================

def delete_from_history(item_id):

    history = load_history()

    updated_history = [
        item
        for item in history
        if item.get("id") != item_id
    ]

    save_history(
        updated_history
    )

    return updated_history


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_history():

    save_history([])


# ============================================================
# DELETE IMAGE FILE
# ============================================================

def delete_image_file(image_path):

    path = Path(image_path)

    if path.exists():

        try:

            path.unlink()

        except OSError:

            pass