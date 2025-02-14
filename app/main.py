# app/main.py
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from app import tasks, utils

app = FastAPI(title="LLM-based Automation Agent")

# The allowed data root is set to /data. You could also make this configurable.
DATA_ROOT = "./data"
os.makedirs(DATA_ROOT, exist_ok=True)  # Ensure the directory exists

@app.post("/run")
async def run_task(task: str = Query(..., description="Plain‑English description of the task")):
    try:
        # Parse and execute the task. This function returns a message.
        message = tasks.parse_and_execute_task(task, data_root=DATA_ROOT)
        return {"status": "success", "message": message}
    except tasks.TaskError as te:
        raise HTTPException(status_code=400, detail=str(te))
    except Exception as e:
        # Log the exception (if using logging) and return a 500.
        raise HTTPException(status_code=500, detail=f"Internal agent error: {str(e)}")

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file within /data")):
    try:
        # Validate that the file path is within the allowed DATA_ROOT
        safe_path = utils.get_safe_path(".", path)
        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
