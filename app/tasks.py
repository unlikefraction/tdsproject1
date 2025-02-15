import os
import re
import subprocess
import json
import sqlite3
import glob
import base64
import datetime
import urllib.request
from app import llm, utils
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import numpy as np
from dateutil.parser import parse
from PIL import Image, ImageEnhance
import easyocr
import re

# Custom exception for task errors
class TaskError(Exception):
    pass

def parse_and_execute_task(task_description: str, data_root: str) -> str:
    """
    Parse the plain-English task description and execute the corresponding task.
    
    Args:
        task_description (str): Description of the task.
        data_root (str): The root directory for input/output files.
    
    Returns:
        str: Success message or error message.
    """
    try:
        # Phase A: Handle predefined tasks with keyword matching
        task_description_lower = task_description.lower()
        print(task_description_lower)
        if "datagen.py" in task_description_lower or "install uv" in task_description_lower:
            return task_a1(data_root)
        elif "prettier" in task_description_lower and "format" in task_description_lower:
            return task_a2(data_root)
        elif "dates.txt" in task_description_lower or "wednesday" in task_description_lower:
            return task_a3(data_root)
        elif "contacts.json" in task_description_lower and "sort" in task_description_lower:
            return task_a4(data_root)
        elif "logs" in task_description_lower and "most recent" in task_description_lower:
            return task_a5(data_root)
        elif "docs" in task_description_lower and "markdown" in task_description_lower:
            return task_a6(data_root)
        elif "email.txt" in task_description_lower and "sender" in task_description_lower:
            return task_a7(data_root)
        elif "credit-card.png" in task_description_lower or "credit card" in task_description_lower:
            return task_a8(data_root)
        elif "comments.txt" in task_description_lower and "similar" in task_description_lower:
            return task_a9(data_root)
        elif "ticket-sales.db" in task_description_lower and "gold" in task_description_lower:
            return task_a10(data_root)
        
        # Phase B: Use LLM to handle open-ended tasks dynamically
        return parse_and_execute_with_llm(task_description, data_root)
    
    except TaskError:
        return "Task not recognized or not supported."
    except Exception as e:
        return f"Error while executing task: {e}"


### Phase A Task Implementations ###

def task_a1(data_root: str) -> str:
    """
    A1. Run datagen.py with the user email.
    We assume that the user email is available in an environment variable, e.g. USER_EMAIL.
    """
    user_email = os.environ.get("USER_EMAIL") or "22f3001874@ds.study.iitm.ac.in"
    os.environ["USER_EMAIL"] = user_email

    datagen_path = os.path.join(os.getcwd(), "datagen.py")
    if not os.path.exists(datagen_path):
        # Download datagen.py from the known URL if it doesn't exist.
        url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        try:
            urllib.request.urlretrieve(url, datagen_path)
        except Exception as e:
            raise TaskError(f"datagen.py not found and download failed: {e}")

    try:
        # Preferably use 'uv' if available. Otherwise, fallback to python.
        cmd = ["uv", "run", datagen_path, user_email]
        subprocess.run(cmd, check=True, timeout=15, env=os.environ.copy())
    except FileNotFoundError:
        cmd = ["python", datagen_path, user_email]
        subprocess.run(cmd, check=True, timeout=15, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        raise TaskError(f"datagen.py execution failed: {e}")
    return "datagen.py executed successfully."

def task_a2(data_root: str) -> str:
    """
    A2. Format /data/format.md using prettier@3.4.2.
    We assume that Node.js and the correct version of prettier are installed.
    """
    file_path = utils.get_safe_path(data_root, "format.md")
    cmd = ["npx", "prettier@3.4.2", "--write", file_path]
    try:
        subprocess.run(cmd, check=True, timeout=15)
    except subprocess.CalledProcessError as e:
        raise TaskError(f"Prettier formatting failed: {e}")
    return "File format.md formatted successfully with prettier@3.4.2."

def task_a3(data_root: str) -> str:
    """
    A3. Count the number of Wednesdays in /data/dates.txt and write the count to /data/dates-wednesdays.txt.
    """
    input_path = utils.get_safe_path(data_root, "dates.txt")
    output_path = utils.get_safe_path(data_root, "dates-wednesdays.txt")
    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dt = parse(line)
                if dt.weekday() == 2:
                    count += 1
            except Exception:
                continue
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(count))
    return f"Found {count} Wednesdays."

def task_a4(data_root: str) -> str:
    """
    A4. Sort contacts in /data/contacts.json by last_name then first_name.
    """
    input_path = utils.get_safe_path(data_root, "contacts.json")
    output_path = utils.get_safe_path(data_root, "contacts-sorted.json")
    with open(input_path, "r", encoding="utf-8") as f:
        contacts = json.load(f)
    contacts_sorted = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contacts_sorted, f, indent=2)
    return "Contacts sorted and written to contacts-sorted.json."

def task_a5(data_root: str) -> str:
    """
    A5. Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt.
    """
    logs_dir = utils.get_safe_path(data_root, "logs")
    log_files = sorted(glob.glob(os.path.join(logs_dir, "*.log")), key=os.path.getmtime, reverse=True)
    selected_files = log_files[:10]
    lines = []
    for file in selected_files:
        with open(file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            lines.append(first_line.rstrip("\n"))
    output_path = utils.get_safe_path(data_root, "logs-recent.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "Processed most recent 10 log files."

def task_a6(data_root: str) -> str:
    """
    A6. Extract the first occurrence of an H1 from each Markdown file under /data/docs/
    and create an index JSON mapping filename to title.
    """
    docs_root = utils.get_safe_path(data_root, "docs")
    index = {}
    for filepath in glob.glob(os.path.join(docs_root, "**", "*.md"), recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    title = line[2:].strip()
                    rel_path = os.path.relpath(filepath, docs_root)
                    index[rel_path] = title
                    break
    output_path = utils.get_safe_path(data_root, "docs/index.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return "Index for markdown titles created."

def task_a7(data_root: str) -> str:
    """
    A7. Use the LLM to extract the sender’s email address from /data/email.txt.
    """
    input_path = utils.get_safe_path(data_root, "email.txt")
    output_path = utils.get_safe_path(data_root, "email-sender.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        email_content = f.read()
    prompt = f"Extract the sender's email address. first reason how to solve it. then return the email address inside <email>...</email>. Extract the sender's email address from the following email message:\n\n{email_content}"
    try:
        result = llm.generate_chat_completion(prompt)
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", result)
        email_extracted = match.group(0) if match else result.strip()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(email_extracted)
        return "Sender's email extracted using LLM."
    except Exception as e:
        raise RuntimeError(f"Failed to extract email using LLM: {e}")

def task_a8(data_root: str) -> str:
    """
    A8. Use EasyOCR to extract the credit card number from /data/credit_card.png.
    This version leverages EasyOCR which often provides better accuracy than Tesseract.
    """
    input_path = utils.get_safe_path(data_root, "credit_card.png")
    output_path = utils.get_safe_path(data_root, "credit-card.txt")
    try:
        # Initialize the EasyOCR reader for English; disable GPU if necessary.
        reader = easyocr.Reader(['en'], gpu=False)
        # Read text from the image. detail=0 returns only the text strings.
        results = reader.readtext(input_path, detail=0)
        # Combine all detected text, then extract only digits.
        combined_text = (" ".join(results)).replace(" ", "")
        print(combined_text)
        credit_card = "".join(re.findall(r"\d+", combined_text))[:16]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(credit_card)
        return "Credit card number extracted using EasyOCR."
    except Exception as e:
        raise RuntimeError(f"Failed to extract credit card number using EasyOCR: {e}")

def task_a9(data_root: str) -> str:
    """
    A9. Find the most similar pair of comments from /data/comments.txt using embeddings.
    """
    input_path = utils.get_safe_path(data_root, "comments.txt")
    output_path = utils.get_safe_path(data_root, "comments-similar.txt")
    with open(input_path, "r", encoding="utf-8") as f:
        comments = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    if len(comments) < 2:
        raise TaskError("Not enough comments to compare.")

    openai_api_base = os.getenv("OPENAI_API_BASE", "https://aiproxy.sanand.workers.dev/openai/v1")
    openai_api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDE4NzRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.n7RQ7yIm8Xck0L6-i_4uphCiEYnyolTZaEjq1LJUV-M"
    try:
        response = httpx.post(
            f"{openai_api_base}/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": comments},
            timeout=15
        )
        response.raise_for_status()
        data_json = response.json()
        embeddings = np.array([item["embedding"] for item in data_json["data"]])
    except Exception as e:
        raise RuntimeError(f"Failed to get embeddings from OpenAI API: {e}")

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)
    i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    similar_pair = sorted([comments[i], comments[j]])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(similar_pair))
    return "Most similar comments found and written to comments-similar.txt."

def task_a10(data_root: str) -> str:
    """
    A10. Query the SQLite DB /data/ticket-sales.db to compute total sales for 'Gold' tickets.
    """
    db_path = utils.get_safe_path(data_root, "ticket-sales.db")
    output_path = utils.get_safe_path(data_root, "ticket-sales-gold.txt")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = 'gold'")
    result = cursor.fetchone()[0]
    conn.close()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(result))
    return "Total sales for Gold tickets calculated."

### Phase B Tasks ###

def execute_generated_code(code: str) -> str:
    """
    Execute the given Python code and return the output.
    """
    try:
        exec_locals = {}
        exec(code, {}, exec_locals)
        return "Code executed successfully."
    except Exception as e:
        raise RuntimeError(f"Error during code execution: {e}")

def install_dependencies(bash_code: str):
    """
    Execute bash commands (typically for installing dependencies).
    """
    try:
        subprocess.run(bash_code, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute bash command: {e}")

def extract_code_blocks(response: str) -> any:
    """
    Extract Python (<python></python>) or bash (<bash></bash>) code blocks from the LLM response.
    
    Args:
        response (str): LLM response containing code blocks.
        
    Returns:
        tuple: Python code, Bash code (if any)
    """
    python_code = re.search(r"<python>(.*?)</python>", response, re.DOTALL)
    bash_code = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
    
    return (python_code.group(1).strip() if python_code else "", 
            bash_code.group(1).strip() if bash_code else "")

def parse_and_execute_with_llm(task_description: str, data_root: str) -> str:
    """
    Use the LLM to reason through the task and generate executable code (Python or Bash).
    
    Args:
        task_description (str): A plain-English description of the task.
        data_root (str): The root directory for input/output files.
        
    Returns:
        str: Success message or error message.
    """
    prompt = (
        f"You are a helpful assistant tasked with automating the following task:\n\n"
        f"{task_description}\n\n"
        "Please explain your thought process step by step and then generate the final code. "
        "Enclose the Python code in <python>...</python> and any program requirements that need to be installed in bash commands in <bash>...</bash>. No need to mention how to run the file. the entire python code will be executed."
        "The Python code should be safe to execute without any harmful commands."
    )

    try:
        response = llm.generate_chat_completion(prompt)
        print("======")
        print(response)
        python_code, bash_code = extract_code_blocks(response)

        if bash_code:
            print("Installing dependencies...")
            install_dependencies(bash_code)

        if python_code:
            print("Executing generated Python code...")
            execute_generated_code(python_code)
            return "Task completed successfully using LLM-generated code."
        else:
            raise RuntimeError("No valid Python code found in the response.")

    except Exception as e:
        raise RuntimeError(f"Failed to complete task: {e}")
