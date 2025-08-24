from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import os
import uuid
import shutil
import traceback
import base64
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO

# AutoViz import
from autoviz.AutoViz_Class import AutoViz_Class

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Data Visualization Generator",
    description="API for automatic data visualization generation using AutoViz",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
app.mount("/static/visualizations", StaticFiles(directory="visualizations"), name="visualizations")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Dashboard template - simplified for visualizations only
dashboard_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Visualization Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background-color: #343a40;
            color: white;
            padding: 15px 0;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .viz-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .viz-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s;
        }
        .viz-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .viz-img {
            width: 100%;
            height: auto;
            border-bottom: 1px solid #eee;
        }
        .viz-content {
            padding: 15px;
        }
        .viz-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #343a40;
        }
        .spinner-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            color: white;
        }
        .spinner-message {
            margin-top: 20px;
            font-size: 20px;
        }
    </style>
</head>
<body>
    <div class="spinner-overlay" id="loadingOverlay">
        <div class="spinner-border text-light" style="width: 5rem; height: 5rem;" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
        <div class="spinner-message">Generating visualizations...</div>
    </div>

    <div class="container">
        <div class="header text-center py-4">
            <h1>Data Visualization Dashboard</h1>
            <p>Interactive visualization powered by AutoViz</p>
        </div>

        {% if not job_id %}
        <!-- Upload Form -->
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h3>Upload Your Dataset</h3>
                    </div>
                    <div class="card-body">
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="dataFile" class="form-label">Select CSV or Excel File</label>
                                <input class="form-control" type="file" id="dataFile" name="file" accept=".csv,.xls,.xlsx" required>
                            </div>
                            <button type="submit" class="btn btn-primary" id="uploadBtn">Upload & Visualize</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        {% else %}
        <!-- Dataset Info -->
        <div class="card mb-4">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h3 class="mb-0">Dataset: {{ filename }}</h3>
                        <p class="text-muted mb-0"><strong>Rows:</strong> {{ rows }} | <strong>Columns:</strong> {{ cols }}</p>
                    </div>
                    <a href="/" class="btn btn-outline-primary">Visualize New Dataset</a>
                </div>
            </div>
        </div>

        <!-- Visualizations Gallery -->
        <h2 class="mt-4 mb-3">Generated Visualizations</h2>
        <div class="viz-gallery">
            {% for viz in visualizations %}
            <div class="viz-card">
                <img src="{{ viz.image_url }}" alt="{{ viz.title }}" class="viz-img">
                <div class="viz-content">
                    <h3 class="viz-title">{{ viz.title }}</h3>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const uploadForm = document.getElementById('uploadForm');
            const loadingOverlay = document.getElementById('loadingOverlay');

            if (uploadForm) {
                uploadForm.addEventListener('submit', async function(e) {
                    e.preventDefault();

                    const formData = new FormData();
                    const fileInput = document.getElementById('dataFile');

                    if (fileInput.files.length === 0) {
                        alert('Please select a file to upload');
                        return;
                    }

                    formData.append('file', fileInput.files[0]);

                    // Show loading overlay
                    loadingOverlay.style.display = 'flex';

                    try {
                        // Upload file
                        const uploadResponse = await fetch('/upload', {
                            method: 'POST',
                            body: formData
                        });

                        if (!uploadResponse.ok) {
                            throw new Error('File upload failed');
                        }

                        const uploadResult = await uploadResponse.json();
                        const jobId = uploadResult.job_id;

                        // Generate visualizations
                        const generateResponse = await fetch(`/generate/${jobId}`, {
                            method: 'POST'
                        });

                        if (!generateResponse.ok) {
                            throw new Error('Visualization generation failed');
                        }

                        // Redirect to dashboard
                        window.location.href = `/dashboard/${jobId}`;
                    } catch (error) {
                        loadingOverlay.style.display = 'none';
                        alert('Error: ' + error.message);
                        console.error(error);
                    }
                });
            }
        });
    </script>
</body>
</html>
"""

# Write template to file
with open("templates/dashboard.html", "w") as f:
    f.write(dashboard_template)

# Job storage
ANALYSIS_JOBS = {}


def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Display the home page with file upload form"""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file (CSV or Excel)"""
    # Generate a unique job ID
    job_id = generate_job_id()

    # Create job entry
    ANALYSIS_JOBS[job_id] = {
        "status": "pending",
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "message": "File uploaded, waiting to generate visualizations",
        "visualizations": []
    }

    # Save file
    file_path = f"uploads/{job_id}_{file.filename}"

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Verify file can be loaded as dataframe
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")

        # Store file path and basic info
        ANALYSIS_JOBS[job_id]["file_path"] = file_path
        ANALYSIS_JOBS[job_id]["shape"] = df.shape
        ANALYSIS_JOBS[job_id]["columns"] = df.columns.tolist()

        return {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "message": "File uploaded successfully, ready to generate visualizations",
            "created_at": ANALYSIS_JOBS[job_id]["created_at"],
            "filename": file.filename,
            "shape": df.shape,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist()
        }

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        # Clean up if file was created
        if os.path.exists(file_path):
            os.remove(file_path)
        # Delete job info
        if job_id in ANALYSIS_JOBS:
            del ANALYSIS_JOBS[job_id]
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/{job_id}")
async def generate_visualizations(
        job_id: str,
        target_column: Optional[str] = Form(None),
        max_rows: Optional[int] = Form(5000),
        max_cols: Optional[int] = Form(30)
):
    """Generate visualizations only (no report)"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = ANALYSIS_JOBS[job_id]

    try:
        # Update job status
        job["status"] = "processing"
        job["progress"] = 20
        job["message"] = "Starting visualization generation"

        # Get file path
        file_path = job["file_path"]

        # Load dataframe
        file_extension = os.path.splitext(job["filename"])[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Update progress
        job["progress"] = 40
        job["message"] = "Starting visualization generation with AutoViz"

        # Create output directory for visualizations
        viz_dir = f"visualizations/{job_id}"
        os.makedirs(viz_dir, exist_ok=True)

        # Initialize AutoViz
        logger.info(f"Starting AutoViz for job {job_id}")
        AV = AutoViz_Class()

        # Generate visualizations
        sep = "," if file_extension == '.csv' else None
        job["progress"] = 50
        job["message"] = "Generating visualizations"

        dft = AV.AutoViz(
            file_path,
            sep=sep,
            depVar=target_column or "",
            dfte=df,
            header=0,
            verbose=2,
            lowess=False,
            chart_format="png",
            max_rows_analyzed=max_rows,
            max_cols_analyzed=max_cols,
            save_plot_dir=viz_dir
        )

        # Get list of generated visualization files
        viz_files = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Generated {len(viz_files)} visualizations for job {job_id}")

        # Update progress
        job["progress"] = 80
        job["message"] = f"Processing {len(viz_files)} generated visualizations"

        # Save visualization info
        visualizations = []
        for viz_file in viz_files:
            file_path = os.path.join(viz_dir, viz_file)
            viz_name = viz_file.replace('.png', '').replace('.jpg', '').replace('_', ' ').title()

            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')

            viz_info = {
                "id": str(uuid.uuid4()),
                "title": viz_name,
                "filename": viz_file,
                "image_path": file_path,
                "image_url": f"/static/visualizations/{job_id}/{viz_file}",
                "base64_image": base64_image
            }

            visualizations.append(viz_info)

        job["visualizations"] = visualizations

        # Update job status
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Visualizations generated successfully"
        job["completed_at"] = datetime.now().isoformat()
        job["viz_count"] = len(visualizations)

        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "visualization_count": len(visualizations),
            "message": f"Successfully generated {len(visualizations)} visualizations",
            "created_at": job["created_at"],
            "completed_at": job["completed_at"],
            "dashboard_url": f"/dashboard/{job_id}"
        }

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        logger.error(traceback.format_exc())
        job["status"] = "failed"
        job["message"] = f"Visualization generation failed: {str(e)}"
        job["progress"] = 0
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get current job status for frontend polling"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = ANALYSIS_JOBS[job_id]

    # Format response for React frontend
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "created_at": job.get("created_at", job["upload_time"]),
        "filename": job["filename"]
    }

    if job["status"] == "completed":
        response["completed_at"] = job.get("completed_at")
        response["visualization_count"] = job.get("viz_count", 0)

    return response


@app.get("/visualizations/{job_id}")
async def get_visualizations(job_id: str):
    """Get all visualizations for a job"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = ANALYSIS_JOBS[job_id]

    if job["status"] != "completed":
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Visualizations not ready yet"
        }

    # Format visualizations for the frontend
    visualization_images = [viz["base64_image"] for viz in job["visualizations"]]

    return {
        "job_id": job_id,
        "status": job["status"],
        "visualization_count": job["viz_count"],
        "visualizations": visualization_images
    }


@app.get("/dashboard/{job_id}", response_class=HTMLResponse)
async def view_dashboard(request: Request, job_id: str):
    """View visualization dashboard"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = ANALYSIS_JOBS[job_id]

    if job["status"] != "completed":
        # Redirect to status page if job is not complete
        return f"""
        <html>
            <head>
                <meta http-equiv="refresh" content="3;url=/dashboard/{job_id}">
                <title>Processing...</title>
                <style>
                    body {{ font-family: Arial; text-align: center; margin-top: 100px; }}
                    .spinner {{ border: 16px solid #f3f3f3; border-top: 16px solid #3498db; border-radius: 50%; width: 80px; height: 80px; animation: spin 2s linear infinite; margin: 0 auto; }}
                    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                </style>
            </head>
            <body>
                <h1>Processing Your Dataset</h1>
                <div class="spinner"></div>
                <p>Current status: {job['status']}</p>
                <p>Progress: {job['progress']}%</p>
                <p>This page will refresh automatically...</p>
            </body>
        </html>
        """

    # Return dashboard template with visualizations
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "job_id": job_id,
            "filename": job["filename"],
            "rows": job["shape"][0],
            "cols": job["shape"][1],
            "visualizations": job["visualizations"]
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("report_generator_api:app", host="0.0.0.0", port=8004, reload=True)