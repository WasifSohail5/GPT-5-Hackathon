
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict, Any, Optional
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Plotter API",
    description="API for intelligent data visualization using GPT-5",
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
app.mount("/static", StaticFiles(directory="plots"), name="static")

# Global settings
SETTINGS = {
    "api_base": os.getenv("GPT5_API_BASE", "https://api.aimlapi.com/v1"),
    "api_key": os.getenv("GPT5_API_KEY", "361a27a60f6b4138982fd15278917fed"),
    "model": os.getenv("GPT5_MODEL", "openai/gpt-5-chat-latest"),
}

# Job storage
PLOT_JOBS = {}

# Request models
class PlotRequest(BaseModel):
    prompt: str
    plot_type: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    title: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class MultiPlotRequest(BaseModel):
    prompts: List[str]
    options: Optional[Dict[str, Any]] = None

def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())

@app.get("/")
async def read_root():
    """Return API info"""
    return {
        "name": "Plotter API",
        "version": "1.0.0",
        "description": "API for intelligent data visualization using GPT-5",
        "endpoints": [
            {"path": "/upload", "method": "POST", "description": "Upload dataset file"},
            {"path": "/columns/{job_id}", "method": "GET", "description": "Get dataset columns"},
            {"path": "/plot/{job_id}", "method": "POST", "description": "Generate a plot based on prompt"},
            {"path": "/plots/{job_id}", "method": "POST", "description": "Generate multiple plots"},
            {"path": "/recommend/{job_id}", "method": "GET", "description": "Get plot recommendations"},
            {"path": "/plot-image/{job_id}/{plot_id}", "method": "GET", "description": "Get plot image"}
        ]
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    """
    Upload a dataset file (CSV or Excel)
    """
    # Generate a unique job ID
    job_id = generate_job_id()
    
    # Create job entry
    PLOT_JOBS[job_id] = {
        "status": "uploading",
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "plots": {}
    }
    
    # Save file
    file_path = f"uploads/{job_id}_{file.filename}"
    
    try:
        # Use custom API key if provided
        if api_key:
            api_key_to_use = api_key
        else:
            api_key_to_use = SETTINGS["api_key"]
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load dataset
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")
        
        # Basic data profiling
        PLOT_JOBS[job_id]["status"] = "ready"
        PLOT_JOBS[job_id]["file_path"] = file_path
        PLOT_JOBS[job_id]["shape"] = df.shape
        PLOT_JOBS[job_id]["columns"] = df.columns.tolist()
        
        # Save dtypes
        dtypes = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtypes[col] = "numeric"
            elif pd.api.types.is_datetime64_dtype(df[col]):
                dtypes[col] = "datetime"
            elif len(df[col].unique()) / len(df[col]) < 0.1:  # Heuristic for categorical
                dtypes[col] = "categorical"
            else:
                dtypes[col] = "text"
        
        PLOT_JOBS[job_id]["dtypes"] = dtypes
        
        # Initialize GPT-5 chat model
        PLOT_JOBS[job_id]["chat"] = ChatOpenAI(
            openai_api_base=SETTINGS["api_base"],
            openai_api_key=api_key_to_use,
            model=SETTINGS["model"]
        )
        
        return {"job_id": job_id, "message": "File uploaded successfully", "columns": df.columns.tolist()}
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/columns/{job_id}")
async def get_columns(job_id: str):
    """Get dataset columns and information"""
    if job_id not in PLOT_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PLOT_JOBS[job_id]
    if job["status"] not in ["ready", "plotting"]:
        raise HTTPException(status_code=400, detail=f"Dataset not ready. Current status: {job['status']}")
    
    # Return column information
    return {
        "columns": job["columns"],
        "dtypes": job["dtypes"],
        "shape": job["shape"]
    }

@app.get("/recommend/{job_id}")
async def recommend_plots(job_id: str):
    """Get plot recommendations for the dataset"""
    if job_id not in PLOT_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PLOT_JOBS[job_id]
    if job["status"] not in ["ready", "plotting"]:
        raise HTTPException(status_code=400, detail=f"Dataset not ready. Current status: {job['status']}")
    
    try:
        # Load the dataset
        file_path = job["file_path"]
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Get basic dataset info
        num_rows, num_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create dataset summary for GPT-5
        column_info = []
        for col in df.columns[:10]:  # Limit to first 10 columns to avoid token limits
            col_type = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info = {
                    "name": col,
                    "type": "numeric",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "null_count": int(df[col].isna().sum())
                }
            else:
                unique_vals = min(5, df[col].nunique())
                sample_values = df[col].dropna().sample(min(5, df[col].dropna().shape[0])).tolist()
                col_info = {
                    "name": col,
                    "type": "categorical" if df[col].nunique() < 10 else "text",
                    "unique_values": int(df[col].nunique()),
                    "sample_values": sample_values,
                    "null_count": int(df[col].isna().sum())
                }
            column_info.append(col_info)
        
        dataset_summary = {
            "rows": num_rows,
            "columns": num_cols,
            "column_info": column_info,
            "numeric_columns": numeric_cols[:10],  # First 10 numeric columns
            "categorical_columns": categorical_cols[:10]  # First 10 categorical columns
        }
        
        # Send to GPT-5 for recommendations
        chat = job["chat"]
        
        system_message = """
        You are an expert data visualization assistant. Based on the dataset summary provided, 
        recommend 5 insightful plots that would reveal interesting patterns or insights. 
        For each plot, provide:
        1. A short title
        2. The plot type (bar, scatter, line, histogram, etc.)
        3. Which columns to use for x and y axes (and other parameters if needed)
        4. A brief justification of why this plot would be insightful
        
        Return your response in the following JSON format:
        {
            "recommendations": [
                {
                    "title": "Plot title",
                    "plot_type": "plot type",
                    "x_column": "x column name",
                    "y_column": "y column name",
                    "additional_params": {}, 
                    "justification": "Why this plot is useful"
                }
            ]
        }
        
        For plots that don't use x/y axes (like pie charts), adapt the format appropriately.
        Focus on creating visualizations that would reveal interesting patterns, relationships, or outliers in the data.
        """
        
        human_message = f"Here's a summary of the dataset I need visualization recommendations for: {json.dumps(dataset_summary)}"
        
        response = chat.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ])
        
        # Extract and parse the JSON response
        try:
            recommendations = json.loads(response.content)
            return recommendations
        except json.JSONDecodeError:
            logger.error("Failed to parse GPT-5 response as JSON")
            return {
                "error": "Failed to parse GPT-5 response as JSON",
                "raw_response": response.content
            }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plot/{job_id}")
async def create_plot(job_id: str, request: PlotRequest):
    """Generate a plot based on prompt or specific parameters"""
    if job_id not in PLOT_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PLOT_JOBS[job_id]
    if job["status"] not in ["ready", "plotting"]:
        raise HTTPException(status_code=400, detail=f"Dataset not ready. Current status: {job['status']}")
    
    try:
        # Load the dataset
        file_path = job["file_path"]
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Generate plot_id
        plot_id = str(uuid.uuid4())
        
        # If prompt is provided, use GPT-5 to interpret it
        if request.prompt and not all([request.plot_type, request.x_column, request.y_column]):
            # Prepare column information for GPT-5
            column_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info = {"name": col, "type": "numeric"}
                else:
                    unique_vals = df[col].nunique()
                    col_info = {"name": col, "type": "categorical" if unique_vals < 10 else "text"}
                column_info.append(col_info)
            
            chat = job["chat"]
            
            system_message = """
            You are an expert data visualization assistant. Based on the user's prompt and dataset columns,
            determine the most appropriate plot type and columns to use. 
            
            Return ONLY a JSON object with the following structure:
            {
                "plot_type": "type of plot (bar, scatter, line, pie, histogram, etc.)",
                "x_column": "column to use for x axis",
                "y_column": "column to use for y axis (if applicable)",
                "additional_params": {
                    "param1": "value1"
                },
                "title": "suggested plot title"
            }
            """
            
            human_message = f"""
            Dataset columns: {json.dumps(column_info)}
            
            User prompt: {request.prompt}
            
            Please determine the appropriate plot type and columns based on this prompt.
            """
            
            response = chat.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ])
            
            try:
                plot_params = json.loads(response.content)
                request.plot_type = plot_params["plot_type"]
                request.x_column = plot_params["x_column"]
                request.y_column = plot_params.get("y_column")  # Some plots don't need y
                request.title = plot_params.get("title", request.prompt)
                additional_params = plot_params.get("additional_params", {})
                if request.options:
                    request.options.update(additional_params)
                else:
                    request.options = additional_params
            except json.JSONDecodeError:
                logger.error("Failed to parse GPT-5 response as JSON")
                raise HTTPException(status_code=500, detail="Failed to interpret plotting request")
        
        # Create plot
        plot_info = await generate_plot(
            df, 
            plot_id, 
            request.plot_type, 
            request.x_column, 
            request.y_column, 
            request.title or "Data Visualization", 
            request.options
        )
        
        # Save plot info
        job["plots"][plot_id] = plot_info
        job["status"] = "plotting"
        
        # Return plot information
        return {
            "plot_id": plot_id,
            "plot_info": plot_info
        }
    
    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plots/{job_id}")
async def create_multiple_plots(job_id: str, request: MultiPlotRequest):
    """Generate multiple plots based on a list of prompts"""
    if job_id not in PLOT_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PLOT_JOBS[job_id]
    if job["status"] not in ["ready", "plotting"]:
        raise HTTPException(status_code=400, detail=f"Dataset not ready. Current status: {job['status']}")
    
    try:
        results = []
        for prompt in request.prompts:
            # Create a single plot request
            plot_request = PlotRequest(
                prompt=prompt,
                options=request.options
            )
            
            # Use the existing endpoint
            result = await create_plot(job_id, plot_request)
            results.append(result)
        
        return {"plots": results}
    
    except Exception as e:
        logger.error(f"Error creating multiple plots: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot-image/{job_id}/{plot_id}")
async def get_plot_image(job_id: str, plot_id: str):
    """Get a plot image by its ID"""
    if job_id not in PLOT_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = PLOT_JOBS[job_id]
    if plot_id not in job["plots"]:
        raise HTTPException(status_code=404, detail="Plot not found")
    
    plot_info = job["plots"][plot_id]
    image_path = plot_info["image_path"]
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Plot image file not found")
    
    return FileResponse(image_path)

async def generate_plot(df, plot_id, plot_type, x_column, y_column, title, options=None):
    """Generate a plot based on specified parameters"""
    if options is None:
        options = {}
    
    # Create an image filename
    image_filename = f"plot_{plot_id}.png"
    image_path = os.path.join("plots", image_filename)
    
    plot_type = plot_type.lower()
    
    try:
        # Use Plotly for interactive visualizations
        fig = None
        
        # Standard plot types
        if plot_type == "bar":
            fig = px.bar(df, x=x_column, y=y_column, title=title, **options)
        
        elif plot_type == "scatter":
            color_col = options.get("color")
            size_col = options.get("size")
            fig = px.scatter(
                df, x=x_column, y=y_column, 
                color=color_col, size=size_col,
                title=title, **options
            )
        
        elif plot_type == "line":
            fig = px.line(df, x=x_column, y=y_column, title=title, **options)
        
        elif plot_type == "histogram":
            fig = px.histogram(df, x=x_column, title=title, **options)
        
        elif plot_type == "box" or plot_type == "boxplot":
            fig = px.box(df, x=x_column, y=y_column, title=title, **options)
        
        elif plot_type == "violin":
            fig = px.violin(df, x=x_column, y=y_column, title=title, **options)
        
        elif plot_type == "pie":
            fig = px.pie(df, names=x_column, values=y_column, title=title, **options)
        
        elif plot_type == "heatmap":
            # For heatmap, we need a pivot table or correlation matrix
            if options.get("correlation", False):
                corr_df = df.corr()
                fig = px.imshow(
                    corr_df, title=title,
                    text_auto=True, aspect="auto", **options
                )
            else:
                # Assume we're doing a pivot table
                pivot_index = x_column
                pivot_columns = options.get("columns", df.columns[1])
                pivot_values = y_column
                pivot_df = df.pivot_table(
                    index=pivot_index, 
                    columns=pivot_columns, 
                    values=pivot_values,
                    aggfunc=options.get("aggfunc", "mean")
                )
                fig = px.imshow(
                    pivot_df, title=title,
                    text_auto=True, aspect="auto", **options
                )
        
        elif plot_type == "area":
            fig = px.area(df, x=x_column, y=y_column, title=title, **options)
        
        elif plot_type == "density":
            fig = px.density_contour(df, x=x_column, y=y_column, title=title, **options)
        
        elif plot_type == "3d_scatter":
            z_column = options.get("z")
            if not z_column:
                raise ValueError("Z column required for 3D scatter plot")
            fig = px.scatter_3d(df, x=x_column, y=y_column, z=z_column, title=title, **options)
        
        else:
            # Default to a bar chart if unknown type
            fig = px.bar(df, x=x_column, y=y_column, title=title)
        
        # Save the image
        if fig:
            fig.write_image(image_path)
            
            # Convert to base64 for embedding
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Return plot information
            return {
                "plot_type": plot_type,
                "x_column": x_column,
                "y_column": y_column,
                "title": title,
                "image_path": image_path,
                "image_url": f"/plot-image/{image_filename}",
                "base64_image": f"data:image/png;base64,{base64_image}",
                "created_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error generating {plot_type} plot: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a simple error image with matplotlib
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title(f"Error: {plot_type} plot")
        plt.savefig(image_path)
        
        # Convert error image to base64
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        return {
            "plot_type": plot_type,
            "error": str(e),
            "title": f"Error: {title}",
            "image_path": image_path,
            "image_url": f"/plot-image/{image_filename}",
            "base64_image": f"data:image/png;base64,{base64_image}",
            "created_at": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("plotter_api:app", host="0.0.0.0", port=8001, reload=True)