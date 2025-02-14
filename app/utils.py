# app/utils.py
import os

def get_safe_path(root: str, relative_path: str) -> str:
    """
    Ensure that the requested path is under the designated root directory.
    """
    # Construct the absolute path
    abs_path = os.path.abspath(os.path.join(root, relative_path.lstrip("/")))
    # Check if the abs_path is a subdirectory of root
    if not abs_path.startswith(os.path.abspath(root)):
        raise ValueError("Access to paths outside the data directory is not allowed.")
    return abs_path
