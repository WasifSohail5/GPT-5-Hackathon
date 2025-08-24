from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import numpy as np
import json
import os
import uuid
import shutil
from typing import Dict, Any, Optional, List
import traceback
from datetime import datetime
import logging
from io import BytesIO, StringIO
import asyncio
from starlette.websockets import WebSocket, WebSocketDisconnect
import openpyxl
# Import our data science agent modules
from data_loader import load_dataset, standardize_missing_values, smart_sample
from data_analyzer import DataAnalyzer
from data_preprocessor import DataPreprocessor
from fastapi.middleware.cors import CORSMiddleware
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Science Agent API",
    description="API for automatic dataset preprocessing using GPT-5",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates


# Global settings
SETTINGS = {
    "api_base": os.getenv("GPT5_API_BASE", "https://api.aimlapi.com/v1"),
    "api_key": os.getenv("GPT5_API_KEY", "361a27a60f6b4138982fd15278917fed"),
    "model": os.getenv("GPT5_MODEL", "openai/gpt-5-chat-latest"),
}

# Job status storage
JOBS = {}
# Active WebSocket connections
ACTIVE_CONNECTIONS = {}


def generate_job_id():
    """Generate a unique job ID"""
    return str(uuid.uuid4())


# Custom StreamHandler for capturing logs for a specific job
class JobLogHandler(logging.StreamHandler):
    def __init__(self, job_id):
        super().__init__(StringIO())
        self.job_id = job_id

    def emit(self, record):
        msg = self.format(record)
        if self.job_id in JOBS:
            if "log_messages" not in JOBS[self.job_id]:
                JOBS[self.job_id]["log_messages"] = []
            JOBS[self.job_id]["log_messages"].append(msg)

            # Send to WebSocket if connected
            if self.job_id in ACTIVE_CONNECTIONS:
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        ACTIVE_CONNECTIONS[self.job_id].send_text(
                            json.dumps({"type": "log", "message": msg})
                        )
                    )
                except RuntimeError:
                    # We're not in an event loop, can't do async
                    pass


@app.get("/")
async def read_root():
    """Return API info"""
    return {
        "name": "Data Science Agent API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/upload", "method": "POST", "description": "Upload dataset file"},
            {"path": "/status/{job_id}", "method": "GET", "description": "Check job status"},
            {"path": "/results/{job_id}", "method": "GET", "description": "Get job results"},
            {"path": "/download/{job_id}/{file_type}", "method": "GET", "description": "Download files"},
            {"path": "/report/{job_id}", "method": "GET", "description": "View HTML report"},
            {"path": "/logs/{job_id}", "method": "GET", "description": "Get job logs"}
        ]
    }


@app.post("/upload")
async def upload_file(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        api_key: Optional[str] = Form(None)
):
    """
    Upload a dataset file (CSV or Excel)
    """
    # Generate a unique job ID
    job_id = generate_job_id()

    # Create job entry
    JOBS[job_id] = {
        "status": "uploading",
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "progress": 0,
        "message": "Uploading file...",
        "log_messages": []
    }

    # Set up job-specific logging
    job_handler = JobLogHandler(job_id)
    job_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    job_logger = logging.getLogger(f"job.{job_id}")
    job_logger.setLevel(logging.INFO)
    job_logger.addHandler(job_handler)
    job_logger.propagate = False  # Don't propagate to root logger

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

        job_logger.info(f"File {file.filename} uploaded successfully")

        JOBS[job_id]["status"] = "uploaded"
        JOBS[job_id]["message"] = "File uploaded successfully. Starting analysis..."
        JOBS[job_id]["file_path"] = file_path

        # Start the analysis and preprocessing in the background
        background_tasks.add_task(
            process_dataset,
            job_id=job_id,
            file_path=file_path,
            api_base=SETTINGS["api_base"],
            api_key=api_key_to_use,
            model=SETTINGS["model"],
            job_logger=job_logger
        )

        return {"job_id": job_id, "status": "processing", "message": "File uploaded and processing started"}

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["message"] = f"Error during upload: {str(e)}"
        job_logger.error(f"Error processing upload: {str(e)}")
        job_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()

    if job_id not in JOBS:
        await websocket.send_text(json.dumps({"error": "Job not found"}))
        await websocket.close()
        return

    # Store the WebSocket connection
    ACTIVE_CONNECTIONS[job_id] = websocket

    try:
        # Send current job status immediately
        job = JOBS[job_id]
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": job["status"],
            "message": job["message"],
            "progress": job["progress"]
        }))

        # Send any existing logs
        if "log_messages" in job:
            for msg in job["log_messages"]:
                await websocket.send_text(json.dumps({
                    "type": "log",
                    "message": msg
                }))

        # Keep connection open until client disconnects
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        if job_id in ACTIVE_CONNECTIONS:
            del ACTIVE_CONNECTIONS[job_id]


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    return JOBS[job_id]


@app.get("/logs/{job_id}")
async def get_job_logs(job_id: str):
    """Get the logs for a job"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = JOBS[job_id].get("log_messages", [])
    return {"job_id": job_id, "logs": logs}


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get the results of a completed job"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]

    if job["status"] != "completed":
        return {"status": job["status"], "message": job["message"]}

    return {
        "status": "completed",
        "dataset_path": job.get("processed_path", ""),
        "analysis_path": job.get("analysis_path", ""),
        "report_path": job.get("report_path", ""),
        "summary": job.get("summary", {})
    }


@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download processed dataset or analysis results"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    if file_type == "dataset":
        file_path = job.get("processed_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Processed dataset not found")
        return FileResponse(path=file_path, filename=os.path.basename(file_path))

    elif file_type == "analysis":
        file_path = job.get("analysis_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Analysis results not found")
        return FileResponse(path=file_path, filename=os.path.basename(file_path))

    elif file_type == "report":
        file_path = job.get("report_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="HTML report not found")
        return FileResponse(path=file_path, filename=os.path.basename(file_path))

    elif file_type == "logs":
        if "log_messages" not in job:
            raise HTTPException(status_code=404, detail="Logs not found")

        log_content = "\n".join(job["log_messages"])
        return StreamingResponse(
            BytesIO(log_content.encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=job_{job_id}_logs.txt"}
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid file type")


@app.get("/report/{job_id}", response_class=HTMLResponse)
async def view_report(job_id: str):
    """View the HTML report directly in the browser"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]

    if job["status"] != "completed":
        return f"<html><body><h1>Job not completed</h1><p>{job['message']}</p></body></html>"

    report_path = job.get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="HTML report not found")

    with open(report_path, "r") as f:
        html_content = f.read()

    return html_content


async def update_job_status(job_id, status, message, progress):
    """Update job status and send WebSocket message if client is connected"""
    if job_id in JOBS:
        JOBS[job_id]["status"] = status
        JOBS[job_id]["message"] = message
        JOBS[job_id]["progress"] = progress

        # Send WebSocket update if connected
        if job_id in ACTIVE_CONNECTIONS:
            try:
                await ACTIVE_CONNECTIONS[job_id].send_text(json.dumps({
                    "type": "status",
                    "status": status,
                    "message": message,
                    "progress": progress
                }))
            except:
                pass  # Client might have disconnected


async def process_dataset(job_id: str, file_path: str, api_base: str, api_key: str, model: str, job_logger):
    """Process the dataset in the background"""
    job = JOBS[job_id]

    try:
        # Step 1: Load the dataset
        await update_job_status(job_id, "loading", "Loading dataset...", 10)
        job_logger.info("Starting to load dataset")

        data = load_dataset(file_path)
        df = data['dataframe']
        metadata = data['metadata']

        job_logger.info(f"Dataset loaded: {metadata['rows']} rows, {metadata['columns']} columns")
        job_logger.info(
            f"Column data types: {', '.join([f'{col}: {dtype}' for col, dtype in list(metadata['dtypes'].items())[:5]])}...")

        # Log missing values stats
        missing_cols = {col: pct for col, pct in metadata['missing_percentage'].items() if pct > 0}
        if missing_cols:
            job_logger.info(f"Detected missing values in {len(missing_cols)} columns")
            for col, pct in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:5]:
                job_logger.info(f"  - {col}: {pct}% missing")
            if len(missing_cols) > 5:
                job_logger.info(f"  - ... and {len(missing_cols) - 5} more columns")

        await update_job_status(job_id, "loading",
                                f"Dataset loaded: {metadata['rows']} rows, {metadata['columns']} columns", 20)

        # Step 2: Analyze the dataset with GPT-5
        await update_job_status(job_id, "analyzing", "Analyzing dataset with GPT-5...", 30)
        job_logger.info("Starting detailed analysis with GPT-5")

        analyzer = DataAnalyzer(api_base=api_base, api_key=api_key, model=model)

        # Initialize time series detection
        job_logger.info("Checking for time series characteristics...")
        time_series_info = analyzer._detect_time_series(df)
        if time_series_info.get('is_time_series'):
            job_logger.info("Time series data detected!")
            for col, info in time_series_info.get('time_columns', {}).items():
                if 'interval' in info:
                    job_logger.info(f"  - Column '{col}' has regular time intervals: {info['interval']}")
                else:
                    job_logger.info(f"  - Column '{col}' has time data but irregular intervals")
        else:
            job_logger.info("No time series characteristics detected")

        # Detect text columns
        text_columns = analyzer._detect_text_columns(df)
        if text_columns:
            job_logger.info(f"Detected {len(text_columns)} text columns that may need special processing")
            for col, info in list(text_columns.items())[:3]:
                job_logger.info(
                    f"  - '{col}': avg length {info['avg_length']:.1f} chars, {info['avg_words']:.1f} words")

        # Detect missing patterns
        job_logger.info("Analyzing missing value patterns...")
        missing_patterns = analyzer._detect_missing_patterns(df)
        if 'empty_rows' in missing_patterns:
            job_logger.info(f"  - Found {missing_patterns['empty_rows']['count']} completely empty rows " +
                            f"({missing_patterns['empty_rows']['percentage']}% of data)")

        if 'correlated_missing' in missing_patterns:
            job_logger.info(f"  - Found correlations in missing values between column pairs")
            for col1, col2, corr in missing_patterns['correlated_missing']['pairs'][:3]:
                job_logger.info(f"    * {col1} and {col2} (correlation: {corr:.2f})")

        job_logger.info("Sending dataset summary to GPT-5 for comprehensive analysis...")
        await update_job_status(job_id, "analyzing",
                                "GPT-5 is analyzing data patterns and recommending preprocessing steps...", 35)

        analysis_results = analyzer.analyze_dataset(df)

        # Save analysis results
        analysis_path = f"results/{job_id}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis_results, f, indent=2)

        job["analysis_path"] = analysis_path

        # Log the analysis results
        job_logger.info("Analysis completed! GPT-5 has provided recommendations")
        if 'dataset_quality_score' in analysis_results:
            job_logger.info(f"Dataset quality score: {analysis_results['dataset_quality_score']}/10")

        job_logger.info("Top preprocessing recommendations:")
        for i, rec in enumerate(analysis_results.get('overall_recommendations', [])[:5]):
            job_logger.info(f"  {i + 1}. {rec}")

        if 'potential_issues' in analysis_results and analysis_results['potential_issues']:
            job_logger.info("Potential issues identified:")
            for issue in analysis_results['potential_issues'][:3]:
                job_logger.info(f"  - {issue}")

        await update_job_status(job_id, "analyzed", "Dataset analysis completed", 50)

        # Step 3: Preprocess the dataset
        await update_job_status(job_id, "preprocessing", "Preprocessing dataset based on analysis...", 60)
        job_logger.info("Starting data preprocessing with advanced techniques")

        preprocessor = DataPreprocessor(analysis_results)

        # Custom preprocessing with detailed logging
        column_analyses = analysis_results.get('column_analyses', {})
        processed_df = df.copy()

        # Handle custom missing values first
        job_logger.info("Standardizing custom missing value indicators...")
        processed_df = preprocessor._handle_custom_missing_values(processed_df)

        # Process columns one by one
        job_logger.info("Processing columns with appropriate preprocessing steps:")

        # Check if we should use parallel processing
        use_parallel = len(df) * len(df.columns) > 100000
        if use_parallel:
            job_logger.info(f"Dataset is large, using parallel processing")
            processed_df = preprocessor.preprocess_dataset_parallel(df)
        else:
            total_cols = len(column_analyses)
            for i, (column, analysis) in enumerate(column_analyses.items()):
                if column not in processed_df.columns:
                    continue

                # Update progress more granularly
                col_progress = 60 + int((i / total_cols) * 20)
                await update_job_status(job_id, "preprocessing", f"Processing column {i + 1}/{total_cols}: {column}",
                                        col_progress)

                job_logger.info(f"Processing column: {column}")

                # Data type conversion
                job_logger.info(f"  - Converting data types for {column}")
                original_dtype = processed_df[column].dtype
                processed_df = preprocessor._convert_data_types(processed_df, column, analysis)
                if processed_df[column].dtype != original_dtype:
                    job_logger.info(f"    * Changed type from {original_dtype} to {processed_df[column].dtype}")

                # Missing values
                if pd.isna(processed_df[column]).any():
                    missing_count = pd.isna(processed_df[column]).sum()
                    missing_pct = (missing_count / len(processed_df)) * 100
                    job_logger.info(
                        f"  - Handling missing values for {column} ({missing_count} missing, {missing_pct:.2f}%)")
                    processed_df = preprocessor._handle_missing_values(processed_df, column, analysis)
                    after_missing = pd.isna(processed_df[column]).sum()
                    if after_missing < missing_count:
                        job_logger.info(f"    * Filled {missing_count - after_missing} missing values")

                # Outliers
                if pd.api.types.is_numeric_dtype(processed_df[column]):
                    job_logger.info(f"  - Checking for outliers in {column}")
                    processed_df = preprocessor._handle_outliers(processed_df, column, analysis)

                # Transformations
                transforms = analysis.get('transformations', '').lower()
                if transforms:
                    job_logger.info(f"  - Applying transformations: {transforms}")
                    col_count_before = len(processed_df.columns)
                    processed_df = preprocessor._apply_transformations(processed_df, column, analysis)
                    new_cols = len(processed_df.columns) - col_count_before
                    if new_cols > 0:
                        job_logger.info(f"    * Created {new_cols} new transformed features")

                # Feature engineering
                if analysis.get('feature_engineering'):
                    job_logger.info(f"  - Engineering features based on {column}")
                    col_count_before = len(processed_df.columns)
                    processed_df = preprocessor._engineer_features(processed_df, column, analysis)
                    new_cols = len(processed_df.columns) - col_count_before
                    if new_cols > 0:
                        job_logger.info(f"    * Created {new_cols} engineered features")

        # Apply overall recommendations
        await update_job_status(job_id, "preprocessing", "Applying overall preprocessing recommendations...", 80)
        job_logger.info("Applying dataset-wide preprocessing recommendations")

        processed_df = preprocessor._apply_overall_recommendations(processed_df)

        # Process text features if any
        if text_columns:
            job_logger.info("Processing text columns with NLP techniques")
            processed_df = preprocessor._process_text_features(processed_df)

        # Create advanced features
        job_logger.info("Creating advanced features and interactions")
        col_count_before = len(processed_df.columns)
        processed_df = preprocessor._create_advanced_features(processed_df)
        new_cols = len(processed_df.columns) - col_count_before
        if new_cols > 0:
            job_logger.info(f"Created {new_cols} advanced features")

        # Step 4: Save processed dataset
        await update_job_status(job_id, "saving", "Saving processed dataset...", 90)

        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)

        processed_path = f"results/{job_id}_processed{ext}"
        if ext.lower() == '.csv':
            processed_df.to_csv(processed_path, index=False)
        elif ext.lower() in ['.xlsx', '.xls']:
            processed_df.to_excel(processed_path, index=False)
        else:
            processed_path = f"results/{job_id}_processed.csv"
            processed_df.to_csv(processed_path, index=False)

        job["processed_path"] = processed_path
        job_logger.info(
            f"Processed dataset saved with shape: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")

        # Compare before and after
        cols_added = set(processed_df.columns) - set(df.columns)
        job_logger.info(f"Added {len(cols_added)} new columns through feature engineering and transformations")

        # Log missing values comparison
        missing_before = df.isna().sum().sum()
        missing_after = processed_df.isna().sum().sum()
        missing_reduction = missing_before - missing_after
        missing_pct = (missing_reduction / missing_before * 100) if missing_before > 0 else 0
        job_logger.info(
            f"Missing values: {missing_before} before → {missing_after} after (reduced by {missing_pct:.1f}%)")

        # Step 5: Generate HTML report
        await update_job_status(job_id, "generating_report", "Generating HTML report...", 95)
        job_logger.info("Generating detailed HTML report with visualizations")

        from main import generate_html_report
        report_path = f"reports/{job_id}_report.html"
        generate_html_report(df, processed_df, analysis_results, report_path)

        job["report_path"] = report_path
        job_logger.info(f"HTML report generated at {report_path}")

        # Prepare summary
        summary = {
            "original_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "processed_shape": {"rows": processed_df.shape[0], "columns": processed_df.shape[1]},
            "missing_values_before": int(df.isna().sum().sum()),
            "missing_values_after": int(processed_df.isna().sum().sum()),
            "quality_score": analysis_results.get("dataset_quality_score", "N/A"),
            "top_recommendations": analysis_results.get("overall_recommendations", [])[:3],
            "columns_added": list(cols_added)[:10],  # Limit to first 10 columns
            "preprocessing_steps": [
                "Standardized missing values",
                "Converted data types",
                "Handled outliers with advanced detection",
                "Applied appropriate transformations",
                "Created engineered features",
                "Processed text data (if present)",
                "Created feature interactions"
            ]
        }

        job["summary"] = summary
        await update_job_status(job_id, "completed", "Processing completed successfully", 100)
        job_logger.info("Data preprocessing completed successfully!")
        job_logger.info(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        job_logger.info(f"Processed dataset: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
        job_logger.info("The dataset is now ready for machine learning!")

    except Exception as e:
        await update_job_status(job_id, "error", f"Error processing dataset: {str(e)}", 0)
        job_logger.error(f"Error processing dataset: {str(e)}")
        job_logger.error(traceback.format_exc())


# Create a better HTML template for the index page with real-time logs
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Data Science Agent</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        body { padding: 20px; }
        .container { max-width: 1000px; }
        .progress { margin: 20px 0; height: 20px; }
        #status-container { display: none; }
        #results-container { display: none; }
        #log-container { 
            height: 300px; 
            overflow-y: auto; 
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 10px;
            font-family: monospace;
            margin-bottom: 20px;
            color: #212529;
        }
        .log-info { color: #0d6efd; }
        .log-error { color: #dc3545; }
        .log-warning { color: #ffc107; }
        .tab-content {
            border: 1px solid #dee2e6;
            border-top: none;
            padding: 15px;
        }
        .nav-tabs {
            margin-bottom: 0;
        }
        .summary-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .comparison-stat {
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .step-info {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Advanced Data Science Agent</h1>
        <p class="lead">Upload a CSV or Excel file to analyze and preprocess with GPT-5 and advanced techniques</p>

        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Upload Dataset</h5>
            </div>
            <div class="card-body">
                <form id="upload-form">
                    <div class="mb-3">
                        <label for="file" class="form-label">Select CSV or Excel file</label>
                        <input type="file" class="form-control" id="file" name="file" accept=".csv,.xlsx,.xls">
                    </div>
                    <div class="mb-3">
                        <label for="api-key" class="form-label">API Key (optional)</label>
                        <input type="text" class="form-control" id="api-key" name="api_key" placeholder="Enter your GPT-5 API key">
                        <div class="form-text">If not provided, the default API key will be used.</div>
                    </div>
                    <button type="submit" class="btn btn-primary">Upload & Process</button>
                </form>
            </div>
        </div>

        <div id="status-container" class="card mb-4">
            <div class="card-header bg-info text-white">
                <h5 class="mb-0">Processing Status</h5>
            </div>
            <div class="card-body">
                <p id="status-message" class="fw-bold">Processing...</p>
                <div class="progress">
                    <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                </div>

                <h5 class="mt-4 mb-2">Real-time Processing Logs:</h5>
                <div id="log-container"></div>
            </div>
        </div>

        <div id="results-container" class="card">
            <div class="card-header bg-success text-white">
                <h5 class="mb-0">Results</h5>
            </div>
            <div class="card-body">
                <ul class="nav nav-tabs" id="resultTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="summary-tab" data-bs-toggle="tab" data-bs-target="#summary" type="button" role="tab">Summary</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="downloads-tab" data-bs-toggle="tab" data-bs-target="#downloads" type="button" role="tab">Downloads</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="logs-tab" data-bs-toggle="tab" data-bs-target="#logs" type="button" role="tab">Full Logs</button>
                    </li>
                </ul>

                <div class="tab-content" id="resultTabsContent">
                    <div class="tab-pane fade show active" id="summary" role="tabpanel">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="summary-box">
                                    <h5>Dataset Transformation</h5>
                                    <div id="shape-comparison" class="comparison-stat"></div>
                                    <div id="missing-comparison" class="comparison-stat"></div>
                                    <div id="quality-score" class="comparison-stat"></div>
                                    <div id="columns-added" class="mt-3"></div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="summary-box">
                                    <h5>Preprocessing Steps Applied</h5>
                                    <div id="preprocessing-steps"></div>
                                </div>

                                <div class="summary-box mt-3">
                                    <h5>GPT-5 Recommendations</h5>
                                    <div id="top-recommendations"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-pane fade" id="downloads" role="tabpanel">
                        <p class="mb-4">Download results and view the detailed report:</p>
                        <div class="list-group">
                            <a id="download-dataset" href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>Download Processed Dataset</strong>
                                    <div class="small text-muted">Get the fully preprocessed dataset ready for machine learning</div>
                                </div>
                                <span class="badge bg-primary rounded-pill">CSV/Excel</span>
                            </a>
                            <a id="download-analysis" href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>Download GPT-5 Analysis</strong>
                                    <div class="small text-muted">Detailed analysis results in JSON format</div>
                                </div>
                                <span class="badge bg-info rounded-pill">JSON</span>
                            </a>
                            <a id="view-report" href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center" target="_blank">
                                <div>
                                    <strong>View Interactive HTML Report</strong>
                                    <div class="small text-muted">Visual comparison of before and after preprocessing</div>
                                </div>
                                <span class="badge bg-success rounded-pill">HTML</span>
                            </a>
                            <a id="download-report" href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>Download HTML Report</strong>
                                    <div class="small text-muted">Save the detailed report for later</div>
                                </div>
                                <span class="badge bg-success rounded-pill">HTML</span>
                            </a>
                            <a id="download-logs" href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>Download Processing Logs</strong>
                                    <div class="small text-muted">Complete log of all preprocessing steps</div>
                                </div>
                                <span class="badge bg-secondary rounded-pill">TXT</span>
                            </a>
                        </div>
                    </div>

                    <div class="tab-pane fade" id="logs" role="tabpanel">
                        <div id="full-log-container" style="white-space: pre-wrap; font-family: monospace; max-height: 500px; overflow-y: auto;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentJobId = null;
        let statusCheckInterval = null;
        let ws = null;

        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData();
            const fileInput = document.getElementById('file');
            const apiKeyInput = document.getElementById('api-key');

            if (!fileInput.files[0]) {
                alert('Please select a file to upload');
                return;
            }

            formData.append('file', fileInput.files[0]);
            if (apiKeyInput.value) {
                formData.append('api_key', apiKeyInput.value);
            }

            // Show status container
            document.getElementById('status-container').style.display = 'block';
            document.getElementById('results-container').style.display = 'none';
            document.getElementById('status-message').textContent = 'Uploading file...';
            document.getElementById('progress-bar').style.width = '5%';
            document.getElementById('log-container').innerHTML = '';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.job_id) {
                    currentJobId = data.job_id;
                    document.getElementById('status-message').textContent = data.message;

                    // Start WebSocket connection
                    connectWebSocket();

                    // Also start polling as fallback
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = setInterval(checkJobStatus, 2000);
                }
            } catch (error) {
                document.getElementById('status-message').textContent = `Error: ${error.message}`;
            }
        });

        function connectWebSocket() {
            if (!currentJobId) return;

            // Close existing connection if any
            if (ws) {
                ws.close();
            }

            // Create new WebSocket connection
            ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/${currentJobId}`);

            ws.onopen = function(e) {
                console.log("WebSocket connection established");
                addLogMessage("WebSocket connection established");

                // Keep connection alive with ping-pong
                setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send("ping");
                    }
                }, 30000);
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === "status") {
                    // Update status display
                    document.getElementById('status-message').textContent = data.message;
                    document.getElementById('progress-bar').style.width = `${data.progress}%`;

                    // If job is completed or has error
                    if (data.status === 'completed') {
                        clearInterval(statusCheckInterval);
                        getJobResults();

                        // Change progress bar color to success
                        document.getElementById('progress-bar').classList.remove('bg-info');
                        document.getElementById('progress-bar').classList.add('bg-success');
                    } else if (data.status === 'error') {
                        clearInterval(statusCheckInterval);
                        document.getElementById('progress-bar').classList.remove('bg-info');
                        document.getElementById('progress-bar').classList.add('bg-danger');
                    }
                } else if (data.type === "log") {
                    addLogMessage(data.message);
                }
            };

            ws.onclose = function(event) {
                if (!event.wasClean) {
                    console.log(`WebSocket connection closed unexpectedly, code=${event.code}`);
                    // Fallback to polling
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = setInterval(checkJobStatus, 2000);
                }
            };

            ws.onerror = function(error) {
                console.error(`WebSocket error: ${error.message}`);
            };
        }

        function addLogMessage(message) {
            const logContainer = document.getElementById('log-container');
            const logElement = document.createElement('div');

            // Simple formatting for different log levels
            if (message.includes('ERROR')) {
                logElement.className = 'log-error';
            } else if (message.includes('WARNING')) {
                logElement.className = 'log-warning';
            } else {
                logElement.className = 'log-info';
            }

            logElement.textContent = message;
            logContainer.appendChild(logElement);

            // Scroll to bottom
            logContainer.scrollTop = logContainer.scrollHeight;

            // Also add to full logs tab
            const fullLogContainer = document.getElementById('full-log-container');
            if (fullLogContainer) {
                fullLogContainer.textContent += message + '\\n';
            }
        }

        async function checkJobStatus() {
            if (!currentJobId) return;

            try {
                const response = await fetch(`/status/${currentJobId}`);
                const data = await response.json();

                // Update status display
                document.getElementById('status-message').textContent = data.message;
                document.getElementById('progress-bar').style.width = `${data.progress}%`;

                // If job is completed or has error
                if (data.status === 'completed') {
                    clearInterval(statusCheckInterval);
                    getJobResults();

                    // Change progress bar color to success
                    document.getElementById('progress-bar').classList.remove('bg-info');
                    document.getElementById('progress-bar').classList.add('bg-success');
                } else if (data.status === 'error') {
                    clearInterval(statusCheckInterval);
                    document.getElementById('progress-bar').classList.remove('bg-info');
                    document.getElementById('progress-bar').classList.add('bg-danger');
                }

                // Get logs if we don't have WebSocket
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    const logsResponse = await fetch(`/logs/${currentJobId}`);
                    const logsData = await logsResponse.json();

                    const logContainer = document.getElementById('log-container');
                    logContainer.innerHTML = '';

                    logsData.logs.forEach(log => {
                        addLogMessage(log);
                    });
                }
            } catch (error) {
                console.error(`Error checking status: ${error.message}`);
            }
        }

        async function getJobResults() {
            if (!currentJobId) return;

            try {
                const response = await fetch(`/results/${currentJobId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    // Show results container
                    document.getElementById('results-container').style.display = 'block';

                    // Set download links
                    document.getElementById('download-dataset').href = `/download/${currentJobId}/dataset`;
                    document.getElementById('download-analysis').href = `/download/${currentJobId}/analysis`;
                    document.getElementById('download-report').href = `/download/${currentJobId}/report`;
                    document.getElementById('download-logs').href = `/download/${currentJobId}/logs`;
                    document.getElementById('view-report').href = `/report/${currentJobId}`;

                    // Populate summary
                    const summary = data.summary;

                    // Dataset shape comparison
                    document.getElementById('shape-comparison').innerHTML = `
                        <strong>Rows:</strong> ${summary.original_shape.rows} → ${summary.processed_shape.rows}<br>
                        <strong>Columns:</strong> ${summary.original_shape.columns} → ${summary.processed_shape.columns}
                        <div class="progress mt-1" style="height: 10px;">
                            <div class="progress-bar bg-info" role="progressbar" style="width: ${(summary.original_shape.columns / summary.processed_shape.columns) * 100}%" 
                                aria-valuenow="${summary.original_shape.columns}" aria-valuemin="0" aria-valuemax="${summary.processed_shape.columns}"></div>
                        </div>
                    `;

                    // Missing values comparison
                    const missingReduction = summary.missing_values_before - summary.missing_values_after;
                    const missingPct = summary.missing_values_before > 0 
                        ? (missingReduction / summary.missing_values_before * 100).toFixed(1) 
                        : 0;

                    document.getElementById('missing-comparison').innerHTML = `
                        <strong>Missing Values:</strong> ${summary.missing_values_before} → ${summary.missing_values_after} 
                        (${missingPct}% reduction)
                    `;

                    // Quality score
                    document.getElementById('quality-score').innerHTML = `
                        <strong>Dataset Quality Score:</strong> ${summary.quality_score}/10
                    `;

                    // Columns added
                    let columnsHtml = `<strong>New Features Added:</strong> ${summary.columns_added.length}`;
                    if (summary.columns_added.length > 0) {
                        columnsHtml += '<ul class="mt-1 mb-0">';
                        summary.columns_added.slice(0, 5).forEach(col => {
                            columnsHtml += `<li>${col}</li>`;
                        });
                        if (summary.columns_added.length > 5) {
                            columnsHtml += `<li>... and ${summary.columns_added.length - 5} more</li>`;
                        }
                        columnsHtml += '</ul>';
                    }
                    document.getElementById('columns-added').innerHTML = columnsHtml;

                    // Preprocessing steps
                    let stepsHtml = '';
                    summary.preprocessing_steps.forEach((step, index) => {
                        stepsHtml += `<div class="step-info"><strong>${index + 1}.</strong> ${step}</div>`;
                    });
                    document.getElementById('preprocessing-steps').innerHTML = stepsHtml;

                    // Top recommendations
                    let recsHtml = '';
                    if (summary.top_recommendations && summary.top_recommendations.length > 0) {
                        summary.top_recommendations.forEach((rec, index) => {
                            recsHtml += `<div class="step-info"><strong>${index + 1}.</strong> ${rec}</div>`;
                        });
                    }
                    document.getElementById('top-recommendations').innerHTML = recsHtml;
                }
            } catch (error) {
                document.getElementById('status-message').textContent = `Error getting results: ${error.message}`;
            }
        }
    </script>
</body>
</html>
""")
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)