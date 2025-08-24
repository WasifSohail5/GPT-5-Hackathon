interface JobStatus {
  job_id: string
  status: "uploaded" | "processing" | "completed" | "error"
  filename: string
  upload_time: string
  viz_count: number
  completion_time?: string
  error?: string
}

interface DataSummary {
  shape: [number, number]
  columns: string[]
  dtypes: Record<string, string>
  missing_values: Record<string, number>
  numeric_stats: Record<
    string,
    {
      min?: number
      max?: number
      mean?: number
      median?: number
      std?: number
      q1?: number
      q3?: number
    }
  >
  categorical_stats: Record<
    string,
    {
      unique_values: number
      top_values: Record<string, number>
    }
  >
  correlation?: Record<string, Record<string, number>>
}

interface Visualization {
  id: string
  title: string
  filename: string
  image_path: string
  image_url: string
  base64_image: string
}

interface ReportData {
  job_id: string
  status: string
  visualization_count: number
  visualizations: Visualization[]
}

interface UploadResponse {
  job_id: string
  message: string
  filename: string
  shape: [number, number]
  rows: number
  columns: number
  column_names: string[]
}

interface GenerateResponse {
  job_id: string
  status: string
  visualization_count: number
  message: string
  report_url: string
}

class ReportGeneratorAPIClient {
  private baseUrl: string

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || process.env.NEXT_PUBLIC_REPORT_API_URL || "http://localhost:8003"
  }

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
        },
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }

      const contentType = response.headers.get("content-type")
      if (contentType && contentType.includes("application/json")) {
        return await response.json()
      } else {
        return (await response.text()) as unknown as T
      }
    } catch (error) {
      console.error(`[v0] Report Generator API request failed:`, error)
      throw error
    }
  }

  private async makeFormRequest<T>(endpoint: string, formData: FormData): Promise<T> {
    return this.makeRequest<T>(endpoint, {
      method: "POST",
      body: formData,
    })
  }

  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append("file", file)

    return this.makeFormRequest<UploadResponse>("/upload", formData)
  }

  async generateAnalysis(
    jobId: string,
    targetColumn?: string,
    maxRows = 5000,
    maxCols = 30,
  ): Promise<GenerateResponse> {
    const formData = new FormData()
    if (targetColumn) {
      formData.append("target_column", targetColumn)
    }
    formData.append("max_rows", maxRows.toString())
    formData.append("max_cols", maxCols.toString())

    return this.makeFormRequest<GenerateResponse>(`/generate/${jobId}`, formData)
  }

  async getJobStatus(jobId: string): Promise<JobStatus> {
    return this.makeRequest<JobStatus>(`/status/${jobId}`)
  }

  async getVisualizations(jobId: string): Promise<ReportData> {
    return this.makeRequest<ReportData>(`/visualizations/${jobId}`)
  }

  async getVisualization(jobId: string, vizId: string): Promise<Visualization> {
    return this.makeRequest<Visualization>(`/visualization/${jobId}/${vizId}`)
  }

  async getReport(jobId: string): Promise<string> {
    return this.makeRequest<string>(`/report/${jobId}`)
  }

  async downloadReport(jobId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/download/report/${jobId}`)
    if (!response.ok) {
      throw new Error(`Failed to download report: ${response.statusText}`)
    }
    return response.blob()
  }

  async getDatasetSummary(jobId: string): Promise<{ job_id: string; summary: DataSummary }> {
    return this.makeRequest<{ job_id: string; summary: DataSummary }>(`/summary/${jobId}`)
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.makeRequest("/")
      return true
    } catch (error) {
      console.error("Report Generator API connection test failed:", error)
      return false
    }
  }
}

// Export singleton instance
export const reportGeneratorAPI = new ReportGeneratorAPIClient()
export default ReportGeneratorAPIClient
export type { JobStatus, DataSummary, Visualization, ReportData, UploadResponse, GenerateResponse }
