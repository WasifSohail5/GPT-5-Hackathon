/**
 * API Configuration Manager
 * Handles dynamic API endpoint configuration and validation
 */

export interface APISettings {
  gpt5ApiKey: string
  dataApiUrl: string
  reportApiUrl: string // Renamed from plotterApiUrl to reportApiUrl
  chatbotApiUrl: string
}

export class APIConfigManager {
  private static instance: APIConfigManager
  private settings: APISettings

  private constructor() {
    this.settings = this.loadSettings()
  }

  public static getInstance(): APIConfigManager {
    if (!APIConfigManager.instance) {
      APIConfigManager.instance = new APIConfigManager()
    }
    return APIConfigManager.instance
  }

  private loadSettings(): APISettings {
    if (typeof window === "undefined") {
      // Server-side defaults
      return {
        gpt5ApiKey: process.env.GPT5_API_KEY || "",
        dataApiUrl: process.env.DATA_API_URL || "http://localhost:8000",
        reportApiUrl: process.env.NEXT_PUBLIC_REPORT_API_URL || "http://localhost:8001", // Updated environment variable name
        chatbotApiUrl: process.env.CHATBOT_API_URL || "http://localhost:8002",
      }
    }

    // Client-side: load from localStorage
    const saved = localStorage.getItem("dataflow-api-settings")
    if (saved) {
      const parsed = JSON.parse(saved)
      return {
        gpt5ApiKey: parsed.gpt5ApiKey || "",
        dataApiUrl: parsed.dataApiUrl || "http://localhost:8000",
        reportApiUrl: parsed.reportApiUrl || parsed.plotterApiUrl || "http://localhost:8001", // Support migration from old plotterApiUrl
        chatbotApiUrl: parsed.chatbotApiUrl || "http://localhost:8002",
      }
    }

    return {
      gpt5ApiKey: "",
      dataApiUrl: "http://localhost:8000",
      reportApiUrl: "http://localhost:8001", // Updated default URL
      chatbotApiUrl: "http://localhost:8002",
    }
  }

  public getSettings(): APISettings {
    return { ...this.settings }
  }

  public updateSettings(newSettings: Partial<APISettings>): void {
    this.settings = { ...this.settings, ...newSettings }

    if (typeof window !== "undefined") {
      localStorage.setItem("dataflow-api-settings", JSON.stringify(this.settings))
    }
  }

  public getDataApiUrl(endpoint = ""): string {
    const baseUrl = this.settings.dataApiUrl.replace(/\/$/, "")
    return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, "")}` : baseUrl
  }

  public getReportApiUrl(endpoint = ""): string {
    // Renamed from getPlotterApiUrl to getReportApiUrl
    const baseUrl = this.settings.reportApiUrl.replace(/\/$/, "")
    return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, "")}` : baseUrl
  }

  public getChatbotApiUrl(endpoint = ""): string {
    const baseUrl = this.settings.chatbotApiUrl.replace(/\/$/, "")
    return endpoint ? `${baseUrl}/${endpoint.replace(/^\//, "")}` : baseUrl
  }

  public getGPT5ApiKey(): string {
    return this.settings.gpt5ApiKey
  }

  public async testConnection(url: string): Promise<boolean> {
    try {
      const response = await fetch(url, {
        method: "GET",
        timeout: 5000,
      } as RequestInit)
      return response.ok
    } catch (error) {
      return false
    }
  }

  public async validateAllConnections(): Promise<{
    dataApi: boolean
    reportApi: boolean // Renamed from plotterApi to reportApi
    chatbotApi: boolean
    gpt5Configured: boolean
  }> {
    const [dataApi, reportApi, chatbotApi] = await Promise.all([
      this.testConnection(this.getDataApiUrl()),
      this.testConnection(this.getReportApiUrl()), // Updated method call
      this.testConnection(this.getChatbotApiUrl()),
    ])

    return {
      dataApi,
      reportApi, // Updated property name
      chatbotApi,
      gpt5Configured: !!this.settings.gpt5ApiKey,
    }
  }
}

// Export singleton instance
export const apiConfig = APIConfigManager.getInstance()

export const getApiConfig = () => apiConfig.getSettings()
