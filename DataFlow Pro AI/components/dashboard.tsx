"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Database,
  MessageSquare,
  BarChart3,
  Settings,
  HelpCircle,
  Info,
  Sparkles,
  ArrowRight,
  Plus,
} from "lucide-react"
import Link from "next/link"
import { DataScienceModule } from "@/components/data-science-module"
import ReportGeneratorModule from "@/components/report-generator-module"
import { AIChatbotModule } from "@/components/ai-chatbot-module"
import { ThemeToggle } from "@/components/theme-toggle"

export function Dashboard() {
  const [activeModule, setActiveModule] = useState<string | null>(null)

  const modules = [
    {
      id: "data-science",
      title: "Data Science Agent",
      description: "Upload CSV/Excel files for automated preprocessing and AI-powered analysis",
      icon: Database,
      status: "active",
      gradient: "from-primary to-secondary",
      features: ["CSV/Excel Upload", "GPT-5 Analysis", "Real-time Progress", "Download Results"],
    },
    {
      id: "ai-chatbot",
      title: "AI Chatbot",
      description: "Intelligent conversational assistant with dataset upload and code execution",
      icon: MessageSquare,
      status: "active",
      gradient: "from-accent to-primary",
      features: ["Dataset Upload", "GPT-5 Conversations", "Code Execution", "Data Insights"],
    },
    {
      id: "report-generator",
      title: "Report Generator",
      description: "Automated comprehensive data analysis reports with visualizations using AutoViz",
      icon: BarChart3,
      status: "active",
      gradient: "from-secondary to-accent",
      features: ["AutoViz Integration", "HTML Reports", "Statistical Analysis", "Export PDF/HTML"],
    },
  ]

  if (activeModule === "data-science") {
    return <DataScienceModule onBack={() => setActiveModule(null)} />
  }

  if (activeModule === "ai-chatbot") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
        {/* Navigation */}
        <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <Button variant="ghost" onClick={() => setActiveModule(null)} className="mr-2">
                  ← Back to Dashboard
                </Button>
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-primary-foreground" />
                  </div>
                  <span className="font-serif font-black text-xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                    DataFlow Pro
                  </span>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <Link href="/settings">
                  <Button variant="ghost" size="sm">
                    <Settings className="w-4 h-4 mr-2" />
                    Settings
                  </Button>
                </Link>
                <ThemeToggle />
              </div>
            </div>
          </div>
        </nav>

        {/* Module Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <AIChatbotModule />
        </div>
      </div>
    )
  }

  if (activeModule === "report-generator") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
        {/* Navigation */}
        <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center space-x-4">
                <Button variant="ghost" onClick={() => setActiveModule(null)} className="mr-2">
                  ← Back to Dashboard
                </Button>
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-primary-foreground" />
                  </div>
                  <span className="font-serif font-black text-xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                    DataFlow Pro
                  </span>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <Link href="/settings">
                  <Button variant="ghost" size="sm">
                    <Settings className="w-4 h-4 mr-2" />
                    Settings
                  </Button>
                </Link>
                <ThemeToggle />
              </div>
            </div>
          </div>
        </nav>

        {/* Module Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <ReportGeneratorModule />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
      {/* Navigation */}
      <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-serif font-black text-xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                DataFlow Pro
              </span>
            </Link>
            <div className="flex items-center space-x-4">
              <Link href="/settings">
                <Button variant="ghost" size="sm">
                  <Settings className="w-4 h-4 mr-2" />
                  Settings
                </Button>
              </Link>
              <Link href="/help">
                <Button variant="ghost" size="sm">
                  <HelpCircle className="w-4 h-4 mr-2" />
                  Help
                </Button>
              </Link>
              <Link href="/about">
                <Button variant="ghost" size="sm">
                  <Info className="w-4 h-4 mr-2" />
                  About
                </Button>
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </nav>

      {/* Dashboard Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-serif font-black text-3xl sm:text-4xl mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            Dashboard
          </h1>
          <p className="text-xl text-muted-foreground">
            Choose a module to get started with your data science workflow
          </p>
        </div>

        {/* Modules Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {modules.map((module) => {
            const IconComponent = module.icon
            const isActive = module.status === "active"

            return (
              <Card
                key={module.id}
                className={`relative border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-xl transition-all duration-300 hover:scale-105 ${
                  isActive ? "cursor-pointer" : "opacity-75"
                }`}
                onClick={() => isActive && setActiveModule(module.id)}
              >
                <CardHeader className="pb-4">
                  <div className="flex items-start justify-between mb-4">
                    <div
                      className={`w-12 h-12 bg-gradient-to-br ${module.gradient} rounded-xl flex items-center justify-center`}
                    >
                      <IconComponent className="w-6 h-6 text-white" />
                    </div>
                    <Badge
                      variant={isActive ? "default" : "secondary"}
                      className={isActive ? "bg-gradient-to-r from-primary to-secondary" : ""}
                    >
                      {isActive ? "Active" : "Coming Soon"}
                    </Badge>
                  </div>
                  <CardTitle className="font-serif font-bold text-xl mb-2">{module.title}</CardTitle>
                  <CardDescription className="text-base leading-relaxed">{module.description}</CardDescription>
                </CardHeader>

                <CardContent className="pt-0">
                  <div className="space-y-2 mb-6">
                    {module.features.map((feature, index) => (
                      <div key={index} className="flex items-center text-sm text-muted-foreground">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mr-3" />
                        {feature}
                      </div>
                    ))}
                  </div>

                  <Button
                    className={`w-full ${
                      isActive
                        ? `bg-gradient-to-r ${module.gradient} hover:opacity-90`
                        : "bg-muted text-muted-foreground cursor-not-allowed"
                    }`}
                    disabled={!isActive}
                  >
                    {isActive ? (
                      <>
                        Launch Module
                        <ArrowRight className="ml-2 w-4 h-4" />
                      </>
                    ) : (
                      "Coming Soon"
                    )}
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Add New Module Card */}
        <Card className="border-2 border-dashed border-border/50 bg-card/30 hover:bg-card/50 transition-all duration-300">
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-muted to-muted-foreground/20 rounded-2xl flex items-center justify-center mb-4">
              <Plus className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="font-serif font-bold text-xl mb-2 text-muted-foreground">More Modules Coming Soon</h3>
            <p className="text-muted-foreground mb-4 max-w-md">
              We're constantly adding new modules to enhance your data science workflow. Stay tuned for exciting
              updates!
            </p>
            <Button variant="outline" className="border-primary/20 hover:bg-primary/5 bg-transparent">
              Request Feature
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
