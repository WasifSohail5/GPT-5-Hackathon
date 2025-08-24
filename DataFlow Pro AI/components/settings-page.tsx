"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ArrowLeft, Palette, Bell, Shield, Sparkles, Save, Server, CheckCircle, AlertCircle } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"
import { useTheme } from "next-themes"

export function SettingsPage() {
  const { theme, setTheme: setGlobalTheme } = useTheme()
  const [notifications, setNotifications] = useState(true)

  const [gpt5ApiKey, setGpt5ApiKey] = useState("")
  const [dataApiUrl, setDataApiUrl] = useState("http://localhost:8000")
  const [reportApiUrl, setReportApiUrl] = useState("http://localhost:8004")
  const [chatbotApiUrl, setChatbotApiUrl] = useState("http://localhost:8002")
  const [apiStatus, setApiStatus] = useState({
    dataApi: "unknown",
    reportApi: "unknown",
    chatbotApi: "unknown",
    gpt5: "unknown",
  })
  const [isSaving, setIsSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState("")

  const testApiConnection = async (url: string, type: string) => {
    try {
      const response = await fetch(url, { method: "GET" })
      return response.ok ? "connected" : "error"
    } catch (error) {
      return "error"
    }
  }

  const checkApiStatuses = async () => {
    const [dataStatus, reportStatus, chatbotStatus] = await Promise.all([
      testApiConnection(dataApiUrl, "data"),
      testApiConnection(reportApiUrl, "report"),
      testApiConnection(chatbotApiUrl, "chatbot"),
    ])

    setApiStatus({
      dataApi: dataStatus,
      reportApi: reportStatus,
      chatbotApi: chatbotStatus,
      gpt5: gpt5ApiKey ? "configured" : "not-configured",
    })
  }

  const saveApiSettings = async () => {
    setIsSaving(true)
    try {
      localStorage.setItem(
        "dataflow-api-settings",
        JSON.stringify({
          gpt5ApiKey,
          dataApiUrl,
          reportApiUrl,
          chatbotApiUrl,
        }),
      )

      await checkApiStatuses()
      setSaveMessage("API settings saved successfully!")
      setTimeout(() => setSaveMessage(""), 3000)
    } catch (error) {
      setSaveMessage("Failed to save API settings")
    } finally {
      setIsSaving(false)
    }
  }

  useEffect(() => {
    const savedSettings = localStorage.getItem("dataflow-api-settings")
    if (savedSettings) {
      const settings = JSON.parse(savedSettings)
      setGpt5ApiKey(settings.gpt5ApiKey || "")
      setDataApiUrl(settings.dataApiUrl || "http://localhost:8000")
      setReportApiUrl(settings.reportApiUrl || "http://localhost:8004")
      setChatbotApiUrl(settings.chatbotApiUrl || "http://localhost:8002")
    }
    checkApiStatuses()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
      {/* Navigation */}
      <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/dashboard">
                <Button variant="ghost" className="mr-4">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Dashboard
                </Button>
              </Link>
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="font-serif font-black text-xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  DataFlow Pro
                </span>
              </div>
            </div>
            <ThemeToggle />
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-serif font-black text-3xl sm:text-4xl mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            Settings
          </h1>
          <p className="text-xl text-muted-foreground">Customize your DataFlow Pro experience</p>
        </div>

        <div className="space-y-8">
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="font-serif font-bold flex items-center">
                <Server className="w-5 h-5 mr-2" />
                Backend Configuration
              </CardTitle>
              <CardDescription>Configure your backend services and API endpoints</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Data Science API */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="data-api-url">Data Science API URL</Label>
                  <div className="flex items-center gap-2">
                    {apiStatus.dataApi === "connected" && <CheckCircle className="w-4 h-4 text-green-500" />}
                    {apiStatus.dataApi === "error" && <AlertCircle className="w-4 h-4 text-red-500" />}
                    <span className="text-xs text-muted-foreground">
                      {apiStatus.dataApi === "connected"
                        ? "Connected"
                        : apiStatus.dataApi === "error"
                          ? "Connection Failed"
                          : "Unknown"}
                    </span>
                  </div>
                </div>
                <Input
                  id="data-api-url"
                  placeholder="http://localhost:8000"
                  value={dataApiUrl}
                  onChange={(e) => setDataApiUrl(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">URL for the main data processing backend (Module 1)</p>
              </div>

              <Separator />

              {/* Chatbot API */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="chatbot-api-url">Chatbot API URL</Label>
                  <div className="flex items-center gap-2">
                    {apiStatus.chatbotApi === "connected" && <CheckCircle className="w-4 h-4 text-green-500" />}
                    {apiStatus.chatbotApi === "error" && <AlertCircle className="w-4 h-4 text-red-500" />}
                    <span className="text-xs text-muted-foreground">
                      {apiStatus.chatbotApi === "connected"
                        ? "Connected"
                        : apiStatus.chatbotApi === "error"
                          ? "Connection Failed"
                          : "Unknown"}
                    </span>
                  </div>
                </div>
                <Input
                  id="chatbot-api-url"
                  placeholder="http://localhost:8002"
                  value={chatbotApiUrl}
                  onChange={(e) => setChatbotApiUrl(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  URL for the AI chatbot and conversation backend (Module 2)
                </p>
              </div>

              <Separator />

              {/* Report Generator API */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="report-api-url">Report Generator API URL</Label>
                  <div className="flex items-center gap-2">
                    {apiStatus.reportApi === "connected" && <CheckCircle className="w-4 h-4 text-green-500" />}
                    {apiStatus.reportApi === "error" && <AlertCircle className="w-4 h-4 text-red-500" />}
                    <span className="text-xs text-muted-foreground">
                      {apiStatus.reportApi === "connected"
                        ? "Connected"
                        : apiStatus.reportApi === "error"
                          ? "Connection Failed"
                          : "Unknown"}
                    </span>
                  </div>
                </div>
                <Input
                  id="report-api-url"
                  placeholder="http://localhost:8004"
                  value={reportApiUrl}
                  onChange={(e) => setReportApiUrl(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  URL for the automated report generation and analysis backend (Module 3)
                </p>
              </div>

              <Separator />

              {/* GPT-5 API Key */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="gpt5-key">GPT-5 API Key</Label>
                  <div className="flex items-center gap-2">
                    {apiStatus.gpt5 === "configured" && <CheckCircle className="w-4 h-4 text-green-500" />}
                    {apiStatus.gpt5 === "not-configured" && <AlertCircle className="w-4 h-4 text-yellow-500" />}
                    <span className="text-xs text-muted-foreground">
                      {apiStatus.gpt5 === "configured" ? "Configured" : "Not Configured"}
                    </span>
                  </div>
                </div>
                <Input
                  id="gpt5-key"
                  type="password"
                  placeholder="Enter your GPT-5 API key"
                  value={gpt5ApiKey}
                  onChange={(e) => setGpt5ApiKey(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  Required for AI-powered analysis, chatbot conversations, and automated report generation
                </p>
              </div>

              <Separator />

              {/* Save API Settings */}
              <div className="flex items-center gap-4">
                <Button
                  onClick={saveApiSettings}
                  disabled={isSaving}
                  className="bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90"
                >
                  {isSaving ? (
                    <>
                      <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4 mr-2" />
                      Save API Settings
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={checkApiStatuses}>
                  Test Connections
                </Button>
              </div>

              {/* Save Message */}
              {saveMessage && (
                <Alert className={saveMessage.includes("success") ? "border-green-500" : "border-red-500"}>
                  <AlertDescription>{saveMessage}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Theme Settings */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="font-serif font-bold flex items-center">
                <Palette className="w-5 h-5 mr-2" />
                Theme & Appearance
              </CardTitle>
              <CardDescription>Customize the look and feel of your workspace</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="theme-select">Color Theme</Label>
                  <p className="text-sm text-muted-foreground">Choose your preferred color scheme</p>
                </div>
                <Select value={theme} onValueChange={setGlobalTheme}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                    <SelectItem value="system">System</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div>
                  <Label>Compact Mode</Label>
                  <p className="text-sm text-muted-foreground">Reduce spacing for more content</p>
                </div>
                <Switch />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Animations</Label>
                  <p className="text-sm text-muted-foreground">Enable smooth transitions and effects</p>
                </div>
                <Switch defaultChecked />
              </div>
            </CardContent>
          </Card>

          {/* Notifications */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="font-serif font-bold flex items-center">
                <Bell className="w-5 h-5 mr-2" />
                Notifications
              </CardTitle>
              <CardDescription>Control how and when you receive notifications</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Processing Complete</Label>
                  <p className="text-sm text-muted-foreground">Get notified when data processing finishes</p>
                </div>
                <Switch checked={notifications} onCheckedChange={setNotifications} />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Error Alerts</Label>
                  <p className="text-sm text-muted-foreground">Receive alerts for processing errors</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Weekly Reports</Label>
                  <p className="text-sm text-muted-foreground">Get weekly usage and insights summary</p>
                </div>
                <Switch />
              </div>
            </CardContent>
          </Card>

          {/* Privacy & Security */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="font-serif font-bold flex items-center">
                <Shield className="w-5 h-5 mr-2" />
                Privacy & Security
              </CardTitle>
              <CardDescription>Manage your data privacy and security preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Data Retention</Label>
                  <p className="text-sm text-muted-foreground">Automatically delete processed files after 30 days</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Analytics</Label>
                  <p className="text-sm text-muted-foreground">Help improve the platform with usage analytics</p>
                </div>
                <Switch defaultChecked />
              </div>

              <Separator />

              <div className="space-y-2">
                <Button variant="outline" className="w-full bg-transparent">
                  Export My Data
                </Button>
                <Button variant="outline" className="w-full text-destructive hover:text-destructive bg-transparent">
                  Delete Account
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
