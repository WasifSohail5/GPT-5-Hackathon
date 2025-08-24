from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
import json
import os
import uuid
import shutil
import traceback
from io import BytesIO
import base64
from pydantic import BaseModel
import logging
from datetime import datetime
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Data Science Assistant API",
    description="API for data science chatbot with GPT-5",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving images
app.mount("/static", StaticFiles(directory="visualizations"), name="static")

# Global settings
API_KEY = os.getenv("GPT5_API_KEY", "361a27a60f6b4138982fd15278917fed")
BASE_URL = os.getenv("GPT5_API_BASE", "https://api.aimlapi.com/v1")
MODEL = os.getenv("GPT5_MODEL", "openai/gpt-5-chat-latest")

# Sessions storage
SESSIONS = {}

# Request models
class MessageRequest(BaseModel):
    message: str
    execute_code: Optional[bool] = False

class ChatMessage(BaseModel):
    role: str
    content: str

class DatasetInfoResponse(BaseModel):
    dataset_name: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    head: List[Dict[str, Any]]
    summary: Dict[str, Any]

def generate_session_id():
    """Generate a unique session ID"""
    return str(uuid.uuid4())

@app.get("/")
async def read_root():
    """Return API info"""
    return {
        "name": "Data Science Assistant API",
        "version": "1.0.0",
        "description": "API for data science chatbot with GPT-5",
        "endpoints": [
            {"path": "/upload", "method": "POST", "description": "Upload dataset file"},
            {"path": "/sessions/{session_id}/info", "method": "GET", "description": "Get dataset info"},
            {"path": "/sessions/{session_id}/message", "method": "POST", "description": "Send message to chatbot"},
            {"path": "/sessions/{session_id}/history", "method": "GET", "description": "Get chat history"},
            {"path": "/sessions/{session_id}/visualize", "method": "POST", "description": "Generate visualization"}
        ]
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    """
    Upload a dataset file (CSV, Excel, or JSON)
    """
    # Generate a unique session ID
    session_id = generate_session_id()
    
    # Create session entry
    SESSIONS[session_id] = {
        "status": "uploading",
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "chat_history": []
    }
    
    # Save file
    file_path = f"uploads/{session_id}_{file.filename}"
    
    try:
        # Use custom API key if provided
        api_key_to_use = api_key if api_key else API_KEY
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load dataset
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Create dataset summary
        dataset_summary = f"""
        Dataset: {file.filename}
        Shape: {df.shape}
        Columns: {df.dtypes.to_dict()}
        Head:\n{df.head(3).to_string()}
        """
        
        # Initialize OpenAI client
        client = OpenAI(
            base_url=BASE_URL,
            api_key=api_key_to_use
        )
        
        # Setup initial chat history with system message
        chat_history = [
            {"role": "system", "content": "You are a professional data scientist. \
            When asked, suggest the best possible visualizations for the dataset summary \
            (scatter plots, histograms, line plots, bar plots, heatmaps, etc). \
            Wait for the user to pick one, then the code will generate it. \
            Provide detailed statistical insights and analysis suggestions. \
            Always be helpful and suggest the most appropriate data science techniques."},
            {"role": "user", "content": f"Here is my dataset summary:\n{dataset_summary}"}
        ]
        
        # Get initial response from GPT-5
        response = client.chat.completions.create(
            model=MODEL,
            messages=chat_history,
            temperature=0.7,
            top_p=0.7
        )
        message = response.choices[0].message.content
        
        # Add assistant's response to chat history
        chat_history.append({"role": "assistant", "content": message})
        
        # Update session information
        SESSIONS[session_id] = {
            "status": "active",
            "filename": file.filename,
            "file_path": file_path,
            "upload_time": datetime.now().isoformat(),
            "chat_history": chat_history,
            "dataframe": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "client": client
        }
        
        # Generate dataset summary stats
        summary_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary_stats[col] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    "missing": int(df[col].isna().sum())
                }
            else:
                unique_values = df[col].nunique()
                summary_stats[col] = {
                    "unique_values": int(unique_values),
                    "missing": int(df[col].isna().sum()),
                    "most_common": df[col].value_counts().head(3).to_dict() if unique_values < 100 else "Too many unique values"
                }
        
        SESSIONS[session_id]["summary_stats"] = summary_stats
        
        # Return session information and initial message
        return {
            "session_id": session_id,
            "message": "File uploaded successfully",
            "dataset_info": {
                "name": file.filename,
                "shape": df.shape,
                "columns": df.columns.tolist()
            },
            "initial_message": message
        }
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up session if it exists
        if session_id in SESSIONS:
            SESSIONS[session_id]["status"] = "error"
            SESSIONS[session_id]["error"] = str(e)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/info", response_model=DatasetInfoResponse)
async def get_session_info(session_id: str):
    """Get information about the dataset in a session"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail=f"Session not active. Current status: {session['status']}")
    
    # Recreate DataFrame from stored records
    df = pd.DataFrame(session["dataframe"])
    
    return {
        "dataset_name": session["filename"],
        "shape": session["shape"],
        "columns": session["columns"],
        "dtypes": session["dtypes"],
        "head": df.head(5).to_dict(orient="records"),
        "summary": session["summary_stats"]
    }

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get the chat history for a session"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    
    # Return chat history excluding the system message
    return {"chat_history": session["chat_history"][1:]}

@app.post("/sessions/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest):
    """Send a message to the chatbot"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail=f"Session not active. Current status: {session['status']}")
    
    try:
        # Add user message to chat history
        session["chat_history"].append({"role": "user", "content": request.message})
        
        # Check if we need to execute code
        execute_code = request.execute_code
        code_output = None
        
        # Look for code in the latest assistant message if execute_code is True
        if execute_code:
            # Find the latest assistant message with code
            for msg in reversed(session["chat_history"]):
                if msg["role"] == "assistant" and "```python" in msg["content"]:
                    # Extract code
                    code = msg["content"].split("```python")[1].split("```")[0].strip()
                    
                    # Execute code and capture output
                    code_output = await execute_python_code(code, session)
                    break
            
            if code_output:
                # Add code execution result to chat history
                session["chat_history"].append({"role": "assistant", "content": f"Code execution result:\n{code_output}"})
                
                # Return the code output without calling GPT-5 again
                return {"message": f"Code execution result:\n{code_output}"}
        
        # Send the conversation to GPT-5
        response = session["client"].chat.completions.create(
            model=MODEL,
            messages=session["chat_history"],
            temperature=0.7,
            top_p=0.7
        )
        message = response.choices[0].message.content
        
        # Add assistant's response to chat history
        session["chat_history"].append({"role": "assistant", "content": message})
        
        return {"message": message}
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/visualize")
async def generate_visualization(
    session_id: str,
    plot_type: str = Form(...),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None),
    title: Optional[str] = Form("Visualization"),
    additional_params: Optional[str] = Form("{}")
):
    """Generate a visualization for the dataset"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = SESSIONS[session_id]
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail=f"Session not active. Current status: {session['status']}")
    
    try:
        # Parse additional parameters
        params = json.loads(additional_params)
        
        # Recreate DataFrame from stored records
        df = pd.DataFrame(session["dataframe"])
        
        # Generate plot ID
        plot_id = f"{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        image_path = f"visualizations/{plot_id}.png"
        
        plt.figure(figsize=(10, 6))
        
        # Create the requested plot
        if plot_type == "scatter":
            if not x_column or not y_column:
                raise HTTPException(status_code=400, detail="Scatter plot requires x and y columns")
            sns.scatterplot(data=df, x=x_column, y=y_column, **params)
            
        elif plot_type == "line":
            if not x_column or not y_column:
                raise HTTPException(status_code=400, detail="Line plot requires x and y columns")
            sns.lineplot(data=df, x=x_column, y=y_column, **params)
            
        elif plot_type == "bar":
            if not x_column:
                raise HTTPException(status_code=400, detail="Bar plot requires x column")
            if y_column:
                sns.barplot(data=df, x=x_column, y=y_column, **params)
            else:
                df[x_column].value_counts().plot(kind="bar", **params)
            
        elif plot_type == "histogram":
            if not x_column:
                raise HTTPException(status_code=400, detail="Histogram requires x column")
            sns.histplot(data=df, x=x_column, **params)
            
        elif plot_type == "boxplot":
            if not x_column and not y_column:
                raise HTTPException(status_code=400, detail="Box plot requires at least one column")
            sns.boxplot(data=df, x=x_column, y=y_column, **params)
            
        elif plot_type == "heatmap":
            # For heatmap, generate correlation matrix if no specific columns
            if "correlation" in params and params["correlation"]:
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", **params)
            else:
                raise HTTPException(status_code=400, detail="For custom heatmap, specify columns and parameters")
            
        elif plot_type == "pairplot":
            # Optional subset of columns
            columns = params.get("columns", None)
            if columns:
                sns.pairplot(df[columns], **params)
            else:
                # If no columns specified, use numeric columns only (limit to 5 columns to avoid overwhelming)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                sns.pairplot(df[numeric_cols], **params)
            
        elif plot_type == "countplot":
            if not x_column:
                raise HTTPException(status_code=400, detail="Count plot requires x column")
            sns.countplot(data=df, x=x_column, **params)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported plot type: {plot_type}")
        
        # Set title
        plt.title(title)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(image_path)
        plt.close()
        
        # Convert to base64 for embedding
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Add visualization to chat history
        session["chat_history"].append({
            "role": "assistant", 
            "content": f"Here's the {plot_type} visualization you requested:\n![{title}](/static/{plot_id}.png)"
        })
        
        return {
            "plot_id": plot_id,
            "plot_type": plot_type,
            "title": title,
            "image_path": image_path,
            "image_url": f"/static/{plot_id}.png",
            "base64_image": f"data:image/png;base64,{base64_image}"
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def execute_python_code(code: str, session: dict) -> str:
    """Execute Python code safely in a restricted environment"""
    try:
        # Get the DataFrame from the session
        df = pd.DataFrame(session["dataframe"])
        
        # Create output capture
        output = BytesIO()
        
        # Create a local namespace with safe imports
        local_namespace = {
            "pd": pd,
            "np": np, 
            "plt": plt,
            "sns": sns,
            "df": df,
            "print": lambda *args, **kwargs: print(*args, **kwargs, file=output)
        }
        
        # Execute code in the restricted environment
        exec(code, {"__builtins__": {}}, local_namespace)
        
        # Capture plot if one was created
        if plt.get_fignums():
            # Generate a unique ID for the plot
            plot_id = f"{session['filename'].split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            image_path = f"visualizations/{plot_id}.png"
            
            # Save the plot
            plt.savefig(image_path)
            plt.close()
            
            # Convert to base64
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Add the plot to the output
            output_str = output.getvalue().decode("utf-8")
            output_str += f"\n\n![Plot](/static/{plot_id}.png)"
            
            return output_str
        
        # Return captured output
        return output.getvalue().decode("utf-8")
    
    except Exception as e:
        return f"Error executing code: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_api:app", host="0.0.0.0", port=8002, reload=True)