import os
from typing import Tuple

def allowed_file(filename: str) -> bool:
    allowed_extensions = {"wav", "mp3", "flac", "m4a"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""

def secure_filename(filename: str) -> str:
    # Simple filename sanitizer
    return os.path.basename(filename).replace(" ", "_").replace("..", "_") 