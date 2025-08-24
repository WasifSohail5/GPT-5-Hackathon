"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  Upload,
  FileText,
  BarChart3,
  TrendingUp,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
} from "lucide-react"
import { toast } from "@/hooks/use-toast"

interface JobStatus {
  job_id: string
  status: "pending" | "processing" | "completed" | "failed"
  progress: number
  message: string
  created_at: string
  completed_at?: string
  visualization_count?: number
}

interface VisualizationData {
  job_id: string
  visualizations: string[]  // Base64 encoded images
}

export default function VisualizationGeneratorModule() {
  const [file, setFile] = useState<File | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [visualizations, setVisualizations] = useState<string[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isPolling, setIsPolling] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      const validTypes = [
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      ]
      if (
        validTypes.includes(selectedFile.type) ||
        selectedFile.name.endsWith(".csv") ||
        selectedFile.name.endsWith(".xlsx") ||
        selectedFile.name.endsWith(".xls")
      ) {
        setFile(selectedFile)
        setJobStatus(null)
        setVisualizations([])
      } else {
        toast({
          title: "Invalid file type",
          description: "Please select a CSV or Excel file.",
          variant: "destructive",
        })
      }
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsUploading(true)
    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004"}/upload`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const result = await response.json()
      setJobStatus({
        job_id: result.job_id,
        status: "pending",
        progress: 0,
        message: "Visualization generation started",
        created_at: result.created_at || new Date().toISOString(),
      })

      // Start generating visualizations
      await generateVisualizations(result.job_id)

    } catch (error) {
      console.error("Upload error:", error)
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "An error occurred during upload.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  const generateVisualizations = async (jobId: string) => {
    try {
      // Generate visualizations
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004"}/generate/${jobId}`, {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error(`Visualization generation failed: ${response.statusText}`)
      }

      const result = await response.json()

      // Start polling for status
      startPolling(jobId)

      toast({
        title: "Processing started",
        description: "Generating visualizations. Please wait...",
      })
    } catch (error) {
      console.error("Generation error:", error)
      setJobStatus(prev =>
        prev ? {
          ...prev,
          status: "failed",
          progress: 0,
          message: error instanceof Error ? error.message : "Visualization generation failed"
        } : null
      )
      toast({
        title: "Generation failed",
        description: error instanceof Error ? error.message : "An error occurred during visualization generation.",
        variant: "destructive",
      })
    }
  }

  const startPolling = (jobId: string) => {
    setIsPolling(true)
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004"}/status/${jobId}`,
        )
        if (!response.ok) {
          throw new Error(`Status check failed: ${response.statusText}`)
        }

        const status = await response.json()
        setJobStatus(status)

        if (status.status === "completed") {
          clearInterval(pollInterval)
          setIsPolling(false)
          await fetchVisualizations(jobId)
          toast({
            title: "Visualizations generated successfully",
            description: `${status.visualization_count || 'Multiple'} visualizations were created for your data.`,
          })
        } else if (status.status === "failed") {
          clearInterval(pollInterval)
          setIsPolling(false)
          toast({
            title: "Visualization generation failed",
            description: status.message || "An error occurred during visualization generation.",
            variant: "destructive",
          })
        }
      } catch (error) {
        console.error("Status polling error:", error)
        clearInterval(pollInterval)
        setIsPolling(false)
      }
    }, 2000)
  }

  const fetchVisualizations = async (jobId: string) => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004"}/visualizations/${jobId}`,
      )
      if (!response.ok) {
        throw new Error(`Failed to fetch visualizations: ${response.statusText}`)
      }

      const data = await response.json()
      setVisualizations(data.visualizations || [])
    } catch (error) {
      console.error("Visualization fetch error:", error)
      toast({
        title: "Failed to load visualizations",
        description: "Could not retrieve the generated visualizations.",
        variant: "destructive",
      })
    }
  }

  const resetModule = () => {
    setFile(null)
    setJobStatus(null)
    setVisualizations([])
    setIsUploading(false)
    setIsPolling(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pending":
      case "processing":
        return <RefreshCw className="h-4 w-4 animate-spin" />
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <RefreshCw className="h-4 w-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "pending":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
      case "processing":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
      case "completed":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
      case "failed":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200"
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-pink-500 to-purple-600 bg-clip-text text-transparent">
            Visualization Generator
          </h2>
          <p className="text-muted-foreground">
            Generate insightful data visualizations automatically with AutoViz
          </p>
        </div>
        {(jobStatus || visualizations.length > 0) && (
          <Button variant="outline" onClick={resetModule}>
            <RefreshCw className="h-4 w-4 mr-2" />
            New Visualization
          </Button>
        )}
      </div>

      {/* Upload Section */}
      {!jobStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Dataset
            </CardTitle>
            <CardDescription>Upload your CSV or Excel file to generate automatic visualizations</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="file-upload">Select File</Label>
              <Input
                id="file-upload"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileSelect}
                ref={fileInputRef}
                className="cursor-pointer"
              />
            </div>

            {file && (
              <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">{file.name}</span>
                  <Badge variant="secondary">{(file.size / 1024 / 1024).toFixed(2)} MB</Badge>
                </div>
                <Button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="bg-gradient-to-r from-pink-500 to-purple-600 hover:from-pink-600 hover:to-purple-700"
                >
                  {isUploading ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <BarChart3 className="h-4 w-4 mr-2" />
                      Generate Visualizations
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Status Section */}
      {jobStatus && jobStatus.status !== "completed" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(jobStatus.status)}
              Visualization Generation Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium">Job ID: {jobStatus.job_id}</p>
                <p className="text-sm text-muted-foreground">{jobStatus.message}</p>
              </div>
              <Badge className={getStatusColor(jobStatus.status)}>
                {jobStatus.status.charAt(0).toUpperCase() + jobStatus.status.slice(1)}
              </Badge>
            </div>

            {jobStatus.status === "processing" && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{jobStatus.progress}%</span>
                </div>
                <Progress value={jobStatus.progress} className="w-full" />
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Visualizations Gallery */}
      {visualizations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Generated Visualizations
            </CardTitle>
            <CardDescription>
              {visualizations.length} visualizations automatically created from your data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {visualizations.map((viz, index) => (
                <Card key={index} className="overflow-hidden">
                  <CardHeader className="py-2 px-4">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Visualization {index + 1}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <img
                      src={`data:image/png;base64,${viz}`}
                      alt={`Visualization ${index + 1}`}
                      className="w-full h-auto"
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}