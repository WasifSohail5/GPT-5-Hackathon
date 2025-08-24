"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Upload,
  Send,
  Bot,
  User,
  Play,
  MessageSquare,
  Database,
  AlertCircle,
  CheckCircle,
  Loader2,
  Plus,
  Paperclip,
  History,
  Trash2,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { chatbotApi, type ChatMessage, type ChatSession, type DatasetInfo } from "@/lib/chatbot-api"

interface ChatbotModuleProps {
  className?: string
}

export function AIChatbotModule({ className }: ChatbotModuleProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [message, setMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chatHistory])

  const clearMessages = () => {
    setError(null)
    setSuccess(null)
  }

  const createNewSession = () => {
    setCurrentSession(null)
    setChatHistory([])
    setDatasetInfo(null)
    clearMessages()
  }

  const switchToSession = (session: ChatSession) => {
    setCurrentSession(session)
    setChatHistory([
      {
        role: "assistant",
        content: `Welcome back! I have your dataset "${session.filename}" loaded and ready for analysis.`,
        timestamp: new Date().toISOString(),
      },
    ])
    setDatasetInfo(session.dataset_info || null)
  }

  const deleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
    if (currentSession?.session_id === sessionId) {
      createNewSession()
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setError(null)
    clearMessages()

    try {
      const response = await chatbotApi.uploadDataset(file)

      const newSession: ChatSession = {
        session_id: response.session_id,
        filename: response.dataset_info.name,
        upload_time: new Date().toISOString(),
        status: "active",
        dataset_info: response.dataset_info,
      }

      setSessions((prev) => [newSession, ...prev])
      setCurrentSession(newSession)

      const initialMessage = `Dataset "${file.name}" uploaded successfully! 

**Dataset Overview:**
- **Shape:** ${response.dataset_info.shape[0]} rows × ${response.dataset_info.shape[1]} columns
- **Columns:** ${response.dataset_info.columns.slice(0, 5).join(", ")}${response.dataset_info.columns.length > 5 ? ` and ${response.dataset_info.columns.length - 5} more` : ""}

I'm ready to help you analyze your data! You can ask me to:
- Generate visualizations
- Perform statistical analysis  
- Create data summaries
- Execute custom Python code

What would you like to explore first?`

      setChatHistory([
        {
          role: "assistant",
          content: initialMessage,
          timestamp: new Date().toISOString(),
        },
      ])

      setSuccess(`Dataset "${file.name}" uploaded successfully!`)

      const info = await chatbotApi.getSessionInfo(response.session_id)
      setDatasetInfo(info)
    } catch (error) {
      console.error("[v0] Upload failed:", error)
      setError(error instanceof Error ? error.message : "Upload failed")
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  const handleSendMessage = async () => {
    if (!message.trim() || !currentSession || isLoading) return

    const userMessage: ChatMessage = {
      role: "user",
      content: message.trim(),
      timestamp: new Date().toISOString(),
    }

    setChatHistory((prev) => [...prev, userMessage])
    setMessage("")
    setIsLoading(true)
    setError(null)

    try {
      const response = await chatbotApi.sendMessage(currentSession.session_id, userMessage.content)

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.message,
        timestamp: new Date().toISOString(),
      }

      setChatHistory((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("[v0] Send message failed:", error)
      setError(error instanceof Error ? error.message : "Failed to send message")
    } finally {
      setIsLoading(false)
    }
  }

  const handleExecuteCode = async () => {
    if (!currentSession || isLoading) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await chatbotApi.sendMessage(currentSession.session_id, "Execute the code", true)

      const executionMessage: ChatMessage = {
        role: "assistant",
        content: response.message,
        timestamp: new Date().toISOString(),
      }

      setChatHistory((prev) => [...prev, executionMessage])
    } catch (error) {
      console.error("[v0] Code execution failed:", error)
      setError(error instanceof Error ? error.message : "Code execution failed")
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const renderMessage = (msg: ChatMessage, index: number) => {
    const isUser = msg.role === "user"
    const hasCode = msg.content.includes("```python")

    return (
      <div key={index} className={`group relative px-4 py-6 hover:bg-muted/30 ${isUser ? "bg-muted/20" : ""}`}>
        <div className="max-w-4xl mx-auto flex gap-4">
          <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center">
            {isUser ? (
              <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                <User className="w-4 h-4 text-secondary-foreground" />
              </div>
            ) : (
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                <Bot className="w-4 h-4 text-primary-foreground" />
              </div>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="prose prose-sm dark:prose-invert max-w-none">
              {msg.content.split("```").map((part, i) => {
                if (i % 2 === 0) {
                  return (
                    <div key={i} className="whitespace-pre-wrap leading-relaxed text-foreground">
                      {part.split("\n").map((line, lineIndex) => {
                        if (line.startsWith("**") && line.endsWith("**")) {
                          return (
                            <div key={lineIndex} className="font-semibold mt-3 mb-1 text-foreground">
                              {line.slice(2, -2)}
                            </div>
                          )
                        }
                        if (line.startsWith("- **") && line.includes(":**")) {
                          const [label, ...rest] = line.slice(4).split(":**")
                          return (
                            <div key={lineIndex} className="ml-2 text-foreground">
                              <span className="font-medium">{label}:</span>
                              {rest.join(":**")}
                            </div>
                          )
                        }
                        return line ? (
                          <div key={lineIndex} className="text-foreground">
                            {line}
                          </div>
                        ) : (
                          <div key={lineIndex} className="h-2"></div>
                        )
                      })}
                    </div>
                  )
                } else {
                  const [lang, ...code] = part.split("\n")
                  return (
                    <div key={i} className="my-4">
                      <div className="bg-background/80 border border-border rounded-lg overflow-hidden">
                        <div className="bg-muted px-4 py-2 text-xs text-muted-foreground border-b border-border">
                          {lang || "code"}
                        </div>
                        <div className="p-4 text-accent font-mono text-sm overflow-x-auto">
                          <pre>{code.join("\n")}</pre>
                        </div>
                      </div>
                    </div>
                  )
                }
              })}
            </div>

            {hasCode && !isUser && (
              <div className="mt-3 flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleExecuteCode}
                  disabled={isLoading}
                  className="text-xs bg-transparent"
                >
                  <Play className="w-3 h-3 mr-1" />
                  Execute Code
                </Button>
              </div>
            )}

            <div className="text-xs text-muted-foreground mt-2">
              {new Date(msg.timestamp || "").toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`flex h-screen bg-background ${className}`}>
      <div
        className={`${sidebarCollapsed ? "w-12" : "w-80"} transition-all duration-300 border-r border-border bg-card flex flex-col relative`}
      >
        <div className="absolute -right-3 top-6 z-10">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-6 h-6 p-0 rounded-full bg-background border-2 shadow-md hover:shadow-lg transition-all"
          >
            {sidebarCollapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
          </Button>
        </div>

        {sidebarCollapsed ? (
          <div className="flex flex-col items-center py-4 space-y-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={createNewSession}
              className="w-8 h-8 p-0 bg-primary hover:bg-primary/90 text-primary-foreground"
              title="New Chat"
            >
              <Plus className="w-4 h-4" />
            </Button>
            <div className="w-6 border-t border-border"></div>
            {sessions.slice(0, 3).map((session) => (
              <Button
                key={session.session_id}
                variant="ghost"
                size="sm"
                onClick={() => switchToSession(session)}
                className={`w-8 h-8 p-0 ${
                  currentSession?.session_id === session.session_id
                    ? "bg-muted text-foreground"
                    : "hover:bg-muted/50 text-muted-foreground"
                }`}
                title={session.filename}
              >
                <Database className="w-4 h-4" />
              </Button>
            ))}
          </div>
        ) : (
          <>
            <div className="p-4 border-b border-border">
              <Button
                onClick={createNewSession}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground shadow-md"
              >
                <Plus className="w-4 h-4 mr-2" />
                New Chat
              </Button>
            </div>

            <ScrollArea className="flex-1">
              <div className="p-2">
                <div className="text-xs font-medium text-muted-foreground mb-2 px-2">Recent Conversations</div>
                {sessions.length === 0 ? (
                  <div className="text-sm text-muted-foreground px-2 py-4">
                    No conversations yet. Upload a dataset to start chatting!
                  </div>
                ) : (
                  <div className="space-y-1">
                    {sessions.map((session) => (
                      <div
                        key={session.session_id}
                        onClick={() => switchToSession(session)}
                        className={`group flex items-center gap-2 p-2 rounded-lg cursor-pointer hover:bg-muted/50 transition-colors ${
                          currentSession?.session_id === session.session_id ? "bg-muted border border-border" : ""
                        }`}
                      >
                        <Database className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate text-foreground">{session.filename}</div>
                          <div className="text-xs text-muted-foreground">
                            {new Date(session.upload_time).toLocaleDateString()}
                          </div>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(e) => deleteSession(session.session_id, e)}
                          className="opacity-0 group-hover:opacity-100 w-6 h-6 p-0 hover:bg-destructive/20"
                        >
                          <Trash2 className="w-3 h-3 text-destructive" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </ScrollArea>
          </>
        )}
      </div>

      <div className="flex-1 flex flex-col">
        <div className="border-b border-border p-4 flex items-center justify-between bg-card/50">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="hover:bg-muted"
            >
              <History className="w-4 h-4" />
            </Button>
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-foreground" />
              <h1 className="text-lg font-semibold text-foreground">
                {currentSession ? currentSession.filename : "AI Data Science Chatbot"}
              </h1>
            </div>
          </div>
          {currentSession && (
            <Badge variant="secondary" className="bg-secondary/20 text-secondary border border-secondary/30">
              <Database className="w-3 h-3 mr-1" />
              Dataset Loaded
            </Badge>
          )}
        </div>

        {error && (
          <div className="p-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        )}

        {success && (
          <div className="p-4">
            <Alert className="border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
              <AlertDescription className="text-green-800 dark:text-green-200">{success}</AlertDescription>
            </Alert>
          </div>
        )}

        <div className="flex-1 overflow-hidden">
          {!currentSession ? (
            <div className="h-full flex items-center justify-center bg-background">
              <div className="text-center space-y-6 max-w-md">
                <div className="w-20 h-20 rounded-full bg-primary flex items-center justify-center mx-auto shadow-lg">
                  <MessageSquare className="w-10 h-10 text-primary-foreground" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold mb-2 text-foreground">Welcome to AI Data Science Chatbot</h2>
                  <p className="text-muted-foreground mb-6">
                    Upload your dataset to start an intelligent conversation about your data
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.xlsx,.xls,.json"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    size="lg"
                    className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg hover:shadow-xl transition-all"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload Dataset
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <ScrollArea className="h-full">
              <div className="divide-y divide-border">
                {chatHistory.map((msg, index) => renderMessage(msg, index))}
                {isLoading && (
                  <div className="px-4 py-6">
                    <div className="max-w-4xl mx-auto flex gap-4">
                      <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                        <Bot className="w-4 h-4 text-primary-foreground" />
                      </div>
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
          )}
        </div>

        {currentSession && (
          <div className="border-t border-border p-4 bg-card/50">
            <div className="max-w-4xl mx-auto">
              <div className="relative flex items-end gap-2 bg-card rounded-xl p-2 border border-border shadow-sm">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="flex-shrink-0 hover:bg-muted"
                >
                  <Paperclip className="w-4 h-4" />
                </Button>
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about your data, request visualizations, or get insights..."
                  disabled={isLoading}
                  className="flex-1 border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-foreground"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={!message.trim() || isLoading}
                  size="sm"
                  className="bg-primary hover:bg-primary/90 text-primary-foreground flex-shrink-0 shadow-sm"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
              <div className="text-xs text-muted-foreground mt-2 text-center">
                Upload files, ask questions, or request data analysis and visualizations
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
