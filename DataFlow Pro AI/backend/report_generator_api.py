from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
import seaborn as sns
from io import BytesIO
import jinja2
import markdown
from pydantic import BaseModel

# AutoViz import
from autoviz.AutoViz_Class import AutoViz_Class

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="Data Analysis Report Generator",
    description="API for automatic data visualization and report generation using AutoViz",
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
app.mount("/static/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/static/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Job storage
ANALYSIS_JOBS = {}

class DataSummary(BaseModel):
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, float]
    numeric_stats: Dict[str, Dict[str, float]]
    categorical_stats: Dict[str, Dict[str, Any]]
    correlation: Optional[Dict[str, Dict[str, float]]] = None

def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())

@app.get("/")
async def read_root():
    """Return API info"""
    return {
        "name": "Data Analysis Report Generator API",
        "version": "1.0.0",
        "description": "API for automatic data visualization and report generation using AutoViz",
        "endpoints": [
            {"path": "/upload", "method": "POST", "description": "Upload dataset file"},
            {"path": "/generate/{job_id}", "method": "POST", "description": "Generate visualizations and report"},
            {"path": "/status/{job_id}", "method": "GET", "description": "Check job status"},
            {"path": "/visualizations/{job_id}", "method": "GET", "description": "Get all visualizations"},
            {"path": "/report/{job_id}", "method": "GET", "description": "Get analysis report in HTML format"},
            {"path": "/download/report/{job_id}", "method": "GET", "description": "Download analysis report as HTML file"},
            {"path": "/summary/{job_id}", "method": "GET", "description": "Get dataset summary statistics"}
        ]
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file (CSV or Excel)"""
    # Generate a unique job ID
    job_id = generate_job_id()
    
    # Create job entry
    ANALYSIS_JOBS[job_id] = {
        "status": "uploaded",
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "visualizations": [],
        "summary": {}
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
            "message": "File uploaded successfully", 
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
async def generate_analysis(
    job_id: str, 
    target_column: Optional[str] = Form(None),
    max_rows: Optional[int] = Form(5000),
    max_cols: Optional[int] = Form(30)
):
    """Generate visualizations and analysis report"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ANALYSIS_JOBS[job_id]
    
    try:
        # Update job status
        job["status"] = "processing"
        
        # Get file path
        file_path = job["file_path"]
        
        # Load dataframe
        file_extension = os.path.splitext(job["filename"])[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Create output directory for visualizations
        viz_dir = f"visualizations/{job_id}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create dataset summary
        summary = create_dataset_summary(df)
        job["summary"] = summary
        
        # Initialize AutoViz
        logger.info(f"Starting AutoViz for job {job_id}")
        AV = AutoViz_Class()
        
        # Generate visualizations
        sep = "," if file_extension == '.csv' else None
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
                "base64_image": f"data:image/png;base64,{base64_image}"
            }
            
            visualizations.append(viz_info)
        
        job["visualizations"] = visualizations
        
        # Generate HTML report
        report_path = f"reports/{job_id}_report.html"
        generate_html_report(df, summary, visualizations, job_id, report_path)
        job["report_path"] = report_path
        
        # Update job status
        job["status"] = "completed"
        job["completion_time"] = datetime.now().isoformat()
        job["viz_count"] = len(visualizations)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "visualization_count": len(visualizations),
            "message": f"Successfully generated {len(visualizations)} visualizations and report",
            "report_url": f"/report/{job_id}"
        }
    
    except Exception as e:
        logger.error(f"Error generating analysis: {str(e)}")
        logger.error(traceback.format_exc())
        job["status"] = "error"
        job["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an analysis job"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ANALYSIS_JOBS[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "upload_time": job["upload_time"],
        "viz_count": job.get("viz_count", 0),
        "completion_time": job.get("completion_time", None),
        "error": job.get("error", None)
    }

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
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "visualization_count": job["viz_count"],
        "visualizations": job["visualizations"]
    }

@app.get("/visualization/{job_id}/{viz_id}")
async def get_visualization(job_id: str, viz_id: str):
    """Get a specific visualization by ID"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ANALYSIS_JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Visualizations not ready yet")
    
    for viz in job["visualizations"]:
        if viz["id"] == viz_id:
            return viz
    
    raise HTTPException(status_code=404, detail="Visualization not found")

@app.get("/report/{job_id}", response_class=HTMLResponse)
async def view_report(job_id: str, request: Request):
    """View analysis report in HTML format"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ANALYSIS_JOBS[job_id]
    
    if job["status"] != "completed":
        return f"<html><body><h1>Analysis not completed</h1><p>Current status: {job['status']}</p></body></html>"
    
    if "report_path" not in job:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Read HTML report
    with open(job["report_path"], "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return html_content

@app.get("/download/report/{job_id}")
async def download_report(job_id: str):
    """Download analysis report as HTML file"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ANALYSIS_JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    if "report_path" not in job:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=job["report_path"], 
        filename=f"analysis_report_{job_id}.html",
        media_type="text/html"
    )

@app.get("/summary/{job_id}")
async def get_dataset_summary(job_id: str):
    """Get dataset summary statistics"""
    if job_id not in ANALYSIS_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ANALYSIS_JOBS[job_id]
    
    if job["status"] != "completed":
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Analysis not completed yet"
        }
    
    if "summary" not in job:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return {
        "job_id": job_id,
        "summary": job["summary"]
    }

def create_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create comprehensive dataset summary"""
    # Basic info
    rows, cols = df.shape
    
    # Missing values
    missing_values = {col: float(df[col].isna().sum() / rows * 100) for col in df.columns}
    
    # Numeric columns stats
    numeric_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_stats[col] = {
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
            "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
            "q1": float(df[col].quantile(0.25)) if not pd.isna(df[col].quantile(0.25)) else None,
            "q3": float(df[col].quantile(0.75)) if not pd.isna(df[col].quantile(0.75)) else None
        }
    
    # Categorical columns stats
    categorical_stats = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[col].value_counts().head(10).to_dict()
        categorical_stats[col] = {
            "unique_values": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()}
        }
    
    # Try to calculate correlation matrix for numeric columns
    correlation = None
    try:
        corr_df = df.select_dtypes(include=[np.number]).corr()
        correlation = {col: corr_df[col].to_dict() for col in corr_df.columns}
    except Exception as e:
        logger.warning(f"Could not compute correlation matrix: {str(e)}")
    
    return {
        "shape": (rows, cols),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": missing_values,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "correlation": correlation
    }

def generate_html_report(df: pd.DataFrame, summary: Dict[str, Any], visualizations: List[Dict[str, Any]], job_id: str, output_path: str):
    """Generate HTML report with dataset details and visualizations"""
    # Create a Jinja2 template for the report
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Analysis Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
            .section { margin-bottom: 40px; }
            .viz-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            .viz-card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; }
            .viz-img { width: 100%; height: auto; margin-bottom: 10px; }
            .viz-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .stat-card { background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
            .tab-content { padding: 20px; border: 1px solid #dee2e6; border-top: none; }
            .corr-heatmap { width: 100%; overflow-x: auto; }
            .corr-table { font-size: 12px; }
            .missing-chart { width: 100%; height: 20px; background-color: #f8f9fa; margin-bottom: 5px; position: relative; }
            .missing-value { background-color: #dc3545; height: 100%; position: absolute; left: 0; top: 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Data Analysis Report</h1>
                <p class="text-muted">Generated on {{ current_time }}</p>
            </div>

            <div class="section">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2>Dataset Overview</h2>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="stat-card">
                                    <h4>Basic Information</h4>
                                    <table class="table">
                                        <tr>
                                            <th>Rows:</th>
                                            <td>{{ summary.shape[0] }}</td>
                                        </tr>
                                        <tr>
                                            <th>Columns:</th>
                                            <td>{{ summary.shape[1] }}</td>
                                        </tr>
                                        <tr>
                                            <th>Numeric Columns:</th>
                                            <td>{{ summary.numeric_stats|length }}</td>
                                        </tr>
                                        <tr>
                                            <th>Categorical Columns:</th>
                                            <td>{{ summary.categorical_stats|length }}</td>
                                        </tr>
                                    </table>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="stat-card">
                                    <h4>Missing Values</h4>
                                    {% for col, pct in summary.missing_values|dictsort(by='value', reverse=True) if pct > 0 %}
                                    <div>
                                        <div class="d-flex justify-content-between">
                                            <span>{{ col }}</span>
                                            <span>{{ "%.1f"|format(pct) }}%</span>
                                        </div>
                                        <div class="missing-chart">
                                            <div class="missing-value" style="width: {{ pct }}%;"></div>
                                        </div>
                                    </div>
                                    {% else %}
                                    <p>No missing values found in the dataset.</p>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2>Column Analysis</h2>
                    </div>
                    <div class="card-body">
                        <ul class="nav nav-tabs" id="columnTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="numeric-tab" data-bs-toggle="tab" data-bs-target="#numeric" type="button" role="tab">Numeric Columns</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="categorical-tab" data-bs-toggle="tab" data-bs-target="#categorical" type="button" role="tab">Categorical Columns</button>
                            </li>
                        </ul>
                        <div class="tab-content" id="columnTabsContent">
                            <div class="tab-pane fade show active" id="numeric" role="tabpanel">
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>Column</th>
                                                <th>Min</th>
                                                <th>Max</th>
                                                <th>Mean</th>
                                                <th>Median</th>
                                                <th>Std Dev</th>
                                                <th>Missing</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for col, stats in summary.numeric_stats.items() %}
                                            <tr>
                                                <td>{{ col }}</td>
                                                <td>{{ "%.2f"|format(stats.min) if stats.min is not none else 'N/A' }}</td>
                                                <td>{{ "%.2f"|format(stats.max) if stats.max is not none else 'N/A' }}</td>
                                                <td>{{ "%.2f"|format(stats.mean) if stats.mean is not none else 'N/A' }}</td>
                                                <td>{{ "%.2f"|format(stats.median) if stats.median is not none else 'N/A' }}</td>
                                                <td>{{ "%.2f"|format(stats.std) if stats.std is not none else 'N/A' }}</td>
                                                <td>{{ "%.1f"|format(summary.missing_values.get(col, 0)) }}%</td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="categorical" role="tabpanel">
                                <div class="row">
                                    {% for col, stats in summary.categorical_stats.items() %}
                                    <div class="col-md-6 mb-4">
                                        <div class="card">
                                            <div class="card-header">
                                                <h5>{{ col }}</h5>
                                                <div class="small text-muted">
                                                    {{ stats.unique_values }} unique values | 
                                                    {{ "%.1f"|format(summary.missing_values.get(col, 0)) }}% missing
                                                </div>
                                            </div>
                                            <div class="card-body">
                                                <table class="table table-sm">
                                                    <thead>
                                                        <tr>
                                                            <th>Value</th>
                                                            <th>Count</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {% for value, count in stats.top_values|dictsort(by='value', reverse=true) %}
                                                        <tr>
                                                            <td>{{ value }}</td>
                                                            <td>{{ count }}</td>
                                                        </tr>
                                                        {% endfor %}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {% if summary.correlation %}
            <div class="section">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2>Correlation Analysis</h2>
                    </div>
                    <div class="card-body">
                        <div class="corr-heatmap">
                            <table class="table table-bordered corr-table">
                                <thead>
                                    <tr>
                                        <th></th>
                                        {% for col in summary.correlation.keys() %}
                                        <th>{{ col }}</th>
                                        {% endfor %}
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for col1, values in summary.correlation.items() %}
                                    <tr>
                                        <th>{{ col1 }}</th>
                                        {% for col2 in summary.correlation.keys() %}
                                        <td style="background-color: rgba({{ 255 * (1-((values[col2] + 1)/2)) }}, {{ 255 * ((values[col2] + 1)/2) }}, 255, 0.7);">
                                            {{ "%.2f"|format(values[col2]) }}
                                        </td>
                                        {% endfor %}
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}

            <div class="section">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2>Data Visualizations</h2>
                    </div>
                    <div class="card-body">
                        <div class="viz-gallery">
                            {% for viz in visualizations %}
                            <div class="viz-card">
                                <h5 class="viz-title">{{ viz.title }}</h5>
                                <img src="{{ viz.image_url }}" alt="{{ viz.title }}" class="viz-img">
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Create Jinja2 template
    template = jinja2.Template(template_str)
    
    # Render the template
    html_content = template.render(
        summary=summary,
        visualizations=visualizations,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {output_path}")
    return output_path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("report_generator_api:app", host="0.0.0.0", port=8003, reload=True)
