import { apiConfig } from "./api-config"

export interface ChatMessage {
  role: "user" | "assistant" | "system"
  content: string
  timestamp?: string
}

export interface ChatSession {
  session_id: string
  filename?: string
  upload_time: string
  status: "active" | "uploading" | "error"
  dataset_info?: {
    name: string
    shape: [number, number]
    columns: string[]
  }
}

export interface DatasetInfo {
  dataset_name: string
  shape: [number, number]
  columns: string[]
  dtypes: Record<string, string>
  head: Record<string, any>[]
  summary: Record<string, any>
}

export interface MessageResponse {
  message: string
}

export interface UploadResponse {
  session_id: string
  message: string
  dataset_info: {
    name: string
    shape: [number, number]
    columns: string[]
  }
  initial_message: string
}

export interface VisualizationResponse {
  plot_id: string
  plot_type: string
  title: string
  image_path: string
  image_url: string
  base64_image: string
}

class ChatbotAPIClient {
  private getBaseUrl(): string {
    return apiConfig.getChatbotApiUrl()
  }

  private getApiKey(): string {
    return apiConfig.getGPT5ApiKey()
  }

  private async makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.getBaseUrl()}${endpoint}`

    try {
      console.log(`[v0] Making request to: ${url}`)
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
      }

      return await response.json()
    } catch (error) {
      console.error(`[v0] Chatbot API request failed:`, error)
      throw error
    }
  }

  private async makeFormRequest<T>(endpoint: string, formData: FormData): Promise<T> {
    const url = `${this.getBaseUrl()}${endpoint}`

    try {
      console.log(`[v0] Making form request to: ${url}`)
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
      }

      return await response.json()
    } catch (error) {
      console.error(`[v0] Chatbot API form request failed:`, error)
      throw error
    }
  }

  async uploadDataset(file: File, apiKey?: string): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append("file", file)

    const keyToUse = apiKey || this.getApiKey()
    if (keyToUse) {
      formData.append("api_key", keyToUse)
    }

    return this.makeFormRequest<UploadResponse>("/upload", formData)
  }

  async getSessionInfo(sessionId: string): Promise<DatasetInfo> {
    return this.makeRequest<DatasetInfo>(`/sessions/${sessionId}/info`)
  }

  async getChatHistory(sessionId: string): Promise<{ chat_history: ChatMessage[] }> {
    return this.makeRequest<{ chat_history: ChatMessage[] }>(`/sessions/${sessionId}/history`)
  }

  async sendMessage(sessionId: string, message: string, executeCode = false): Promise<MessageResponse> {
    return this.makeRequest<MessageResponse>(`/sessions/${sessionId}/message`, {
      method: "POST",
      body: JSON.stringify({
        message,
        execute_code: executeCode,
      }),
    })
  }

  async generateVisualization(
    sessionId: string,
    plotType: string,
    options: {
      xColumn?: string
      yColumn?: string
      title?: string
      additionalParams?: Record<string, any>
    } = {},
  ): Promise<VisualizationResponse> {
    const formData = new FormData()
    formData.append("plot_type", plotType)

    if (options.xColumn) {
      formData.append("x_column", options.xColumn)
    }
    if (options.yColumn) {
      formData.append("y_column", options.yColumn)
    }
    if (options.title) {
      formData.append("title", options.title)
    }
    if (options.additionalParams) {
      formData.append("additional_params", JSON.stringify(options.additionalParams))
    }

    return this.makeFormRequest<VisualizationResponse>(`/sessions/${sessionId}/visualize`, formData)
  }

  async testConnection(): Promise<{ status: string; message: string }> {
    try {
      const response = await this.makeRequest<any>("/")
      return {
        status: "success",
        message: `Connected to ${response.name || "Chatbot API"} ${response.version ? `v${response.version}` : ""}`,
      }
    } catch (error) {
      return {
        status: "error",
        message: error instanceof Error ? error.message : "Connection failed",
      }
    }
  }
}

// Export singleton instance
export const chatbotApi = new ChatbotAPIClient()

// Export utility functions
export const formatChatMessage = (message: ChatMessage): ChatMessage => ({
  ...message,
  timestamp: message.timestamp || new Date().toISOString(),
})

export const extractCodeFromMessage = (content: string): string | null => {
  const codeMatch = content.match(/```python\n([\s\S]*?)\n```/)
  return codeMatch ? codeMatch[1].trim() : null
}

export const hasCodeInMessage = (content: string): boolean => {
  return content.includes("```python")
}

export const formatDatasetInfo = (info: DatasetInfo): string => {
  return `Dataset: ${info.dataset_name}
Shape: ${info.shape[0]} rows × ${info.shape[1]} columns
Columns: ${info.columns.join(", ")}`
}
