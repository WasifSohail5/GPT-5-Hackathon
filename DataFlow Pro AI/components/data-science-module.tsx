"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  ArrowLeft,
  Upload,
  FileText,
  Download,
  Eye,
  Loader2,
  CheckCircle,
  AlertCircle,
  Database,
  Sparkles,
} from "lucide-react"

interface DataScienceModuleProps {
  onBack: () => void
}

interface JobStatus {
  status: string
  message: string
  progress: number
  filename?: string
  log_messages?: string[]
}

interface JobResults {
  status: string
  dataset_path?: string
  analysis_path?: string
  report_path?: string
  summary?: any
}

export function DataScienceModule({ onBack }: DataScienceModuleProps) {
  const [file, setFile] = useState<File | null>(null)
  const [apiKey, setApiKey] = useState("")
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [jobResults, setJobResults] = useState<JobResults | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (jobId && !wsRef.current) {
      const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`)
      wsRef.current = ws

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.type === "status") {
          setJobStatus({
            status: data.status,
            message: data.message,
            progress: data.progress,
          })
        } else if (data.type === "log") {
          setLogs((prev) => [...prev, data.message])
        }
      }

      ws.onclose = () => {
        wsRef.current = null
      }

      return () => {
        ws.close()
      }
    }
  }, [jobId])

  // Poll for job completion and results
  useEffect(() => {
    if (jobId && jobStatus?.status === "completed") {
      fetchResults()
    }
  }, [jobId, jobStatus?.status])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsUploading(true)
    setLogs([])

    const formData = new FormData()
    formData.append("file", file)
    if (apiKey) {
      formData.append("api_key", apiKey)
    }

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()

      if (response.ok) {
        setJobId(result.job_id)
        setJobStatus({
          status: result.status,
          message: result.message,
          progress: 0,
        })
      } else {
        console.error("Upload failed:", result)
      }
    } catch (error) {
      console.error("Upload error:", error)
    } finally {
      setIsUploading(false)
    }
  }

  const fetchResults = async () => {
    if (!jobId) return

    try {
      const response = await fetch(`http://localhost:8000/results/${jobId}`)
      const results = await response.json()
      setJobResults(results)
    } catch (error) {
      console.error("Failed to fetch results:", error)
    }
  }

  const handleDownload = async (fileType: string) => {
    if (!jobId) return

    try {
      const response = await fetch(`http://localhost:8000/download/${jobId}/${fileType}`)
      const blob = await response.blob()

      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${jobId}_${fileType}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error("Download failed:", error)
    }
  }

  const handleViewReport = () => {
    if (jobId) {
      window.open(`http://localhost:8000/report/${jobId}`, "_blank")
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-500"
      case "error":
        return "bg-red-500"
      case "processing":
      case "analyzing":
      case "preprocessing":
        return "bg-primary"
      default:
        return "bg-muted-foreground"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case "error":
        return <AlertCircle className="w-5 h-5 text-red-500" />
      case "processing":
      case "analyzing":
      case "preprocessing":
        return <Loader2 className="w-5 h-5 text-primary animate-spin" />
      default:
        return <Database className="w-5 h-5 text-muted-foreground" />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
      {/* Navigation */}
      <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-16">
            <Button variant="ghost" onClick={onBack} className="mr-4">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </Button>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                <Database className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-serif font-black text-xl">Data Science Agent</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="font-serif font-bold flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Upload Dataset
              </CardTitle>
              <CardDescription>Upload your CSV or Excel file for automated preprocessing and analysis</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <Label htmlFor="file-upload">Dataset File</Label>
                <Input
                  id="file-upload"
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleFileChange}
                  className="mt-2"
                />
                {file && (
                  <p className="text-sm text-muted-foreground mt-2">
                    Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </p>
                )}
              </div>

              <div>
                <Label htmlFor="api-key">API Key (Optional)</Label>
                <Input
                  id="api-key"
                  type="password"
                  placeholder="Enter your GPT-5 API key"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="mt-2"
                />
                <p className="text-sm text-muted-foreground mt-1">Leave empty to use default API key</p>
              </div>

              <Button
                onClick={handleUpload}
                disabled={!file || isUploading}
                className="w-full bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Start Analysis
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Status Section */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="font-serif font-bold flex items-center">
                <Sparkles className="w-5 h-5 mr-2" />
                Processing Status
              </CardTitle>
              <CardDescription>Real-time updates on your data processing job</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {jobStatus ? (
                <>
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(jobStatus.status)}
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="outline" className="capitalize">
                          {jobStatus.status}
                        </Badge>
                        <span className="text-sm text-muted-foreground">{jobStatus.progress}%</span>
                      </div>
                      <Progress value={jobStatus.progress} className="mb-2" />
                      <p className="text-sm text-muted-foreground">{jobStatus.message}</p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload a file to start processing</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          {jobResults && jobResults.status === "completed" && (
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm lg:col-span-2">
              <CardHeader>
                <CardTitle className="font-serif font-bold flex items-center">
                  <CheckCircle className="w-5 h-5 mr-2 text-green-500" />
                  Results & Downloads
                </CardTitle>
                <CardDescription>Your analysis is complete. Download the results below.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Button
                    variant="outline"
                    onClick={() => handleDownload("dataset")}
                    className="flex items-center justify-center"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Processed Dataset
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleDownload("analysis")}
                    className="flex items-center justify-center"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Analysis JSON
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleViewReport}
                    className="flex items-center justify-center bg-transparent"
                  >
                    <Eye className="w-4 h-4 mr-2" />
                    View Report
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleDownload("logs")}
                    className="flex items-center justify-center"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Logs
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Logs Section */}
          {logs.length > 0 && (
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm lg:col-span-2">
              <CardHeader>
                <CardTitle className="font-serif font-bold">Processing Logs</CardTitle>
                <CardDescription>Real-time logs from the data processing pipeline</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="bg-muted/50 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
                  {logs.map((log, index) => (
                    <div key={index} className="mb-1 text-muted-foreground">
                      {log}
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
