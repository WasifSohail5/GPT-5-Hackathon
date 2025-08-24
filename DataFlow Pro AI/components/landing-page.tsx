"use client"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowRight, Database, MessageSquare, BarChart3, Sparkles, Zap, Shield, Users } from "lucide-react"
import Link from "next/link"
import { ThemeToggle } from "@/components/theme-toggle"

export function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
      {/* Navigation */}
      <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-serif font-black text-xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                DataFlow Pro
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/dashboard">
                <Button variant="ghost" className="text-muted-foreground hover:text-foreground">
                  Dashboard
                </Button>
              </Link>
              <Link href="/help">
                <Button variant="ghost" className="text-muted-foreground hover:text-foreground">
                  Help
                </Button>
              </Link>
              <ThemeToggle />
              <Link href="/dashboard">
                <Button className="bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90">
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <Badge
            variant="secondary"
            className="mb-6 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary border-primary/20"
          >
            <Zap className="w-4 h-4 mr-2" />
            AI-Powered Data Science Platform
          </Badge>

          <h1 className="font-serif font-black text-4xl sm:text-6xl lg:text-7xl mb-6 bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent leading-tight">
            Transform Your Data
            <br />
            Into Insights
          </h1>

          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
            Professional data science platform with automated preprocessing, AI-powered analysis, and intelligent
            chatbot assistance. Turn complex datasets into actionable insights in minutes.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Link href="/dashboard">
              <Button
                size="lg"
                className="bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90 text-lg px-8 py-6"
              >
                Explore Modules
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
            </Link>
            <Button
              variant="outline"
              size="lg"
              className="text-lg px-8 py-6 border-primary/20 hover:bg-primary/5 bg-transparent"
            >
              Watch Demo
            </Button>
          </div>

          {/* Feature Preview */}
          <div className="relative max-w-5xl mx-auto">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 blur-3xl rounded-full"></div>
            <div className="relative bg-card/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8 shadow-2xl">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-primary to-secondary rounded-xl mx-auto mb-4 flex items-center justify-center">
                    <Database className="w-6 h-6 text-primary-foreground" />
                  </div>
                  <h3 className="font-serif font-bold text-lg mb-2">Smart Processing</h3>
                  <p className="text-muted-foreground text-sm">
                    Automated data cleaning and preprocessing with AI recommendations
                  </p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-accent to-primary rounded-xl mx-auto mb-4 flex items-center justify-center">
                    <MessageSquare className="w-6 h-6 text-accent-foreground" />
                  </div>
                  <h3 className="font-serif font-bold text-lg mb-2">AI Assistant</h3>
                  <p className="text-muted-foreground text-sm">
                    Intelligent chatbot for data analysis guidance and insights
                  </p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-secondary to-accent rounded-xl mx-auto mb-4 flex items-center justify-center">
                    <BarChart3 className="w-6 h-6 text-secondary-foreground" />
                  </div>
                  <h3 className="font-serif font-bold text-lg mb-2">Advanced Analytics</h3>
                  <p className="text-muted-foreground text-sm">
                    Comprehensive reports with visualizations and actionable insights
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-transparent to-card/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="font-serif font-black text-3xl sm:text-4xl mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Powerful Features
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Everything you need for professional data science workflows
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300 hover:scale-105">
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-primary to-secondary rounded-xl mb-4 flex items-center justify-center">
                  <Database className="w-6 h-6 text-primary-foreground" />
                </div>
                <CardTitle className="font-serif font-bold">Data Preprocessing</CardTitle>
                <CardDescription>
                  Automated data cleaning, transformation, and quality assessment with GPT-5 powered recommendations
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300 hover:scale-105">
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-accent to-primary rounded-xl mb-4 flex items-center justify-center">
                  <MessageSquare className="w-6 h-6 text-accent-foreground" />
                </div>
                <CardTitle className="font-serif font-bold">AI Chatbot</CardTitle>
                <CardDescription>
                  Intelligent conversational assistant for data analysis guidance, insights, and recommendations
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300 hover:scale-105">
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-secondary to-accent rounded-xl mb-4 flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-secondary-foreground" />
                </div>
                <CardTitle className="font-serif font-bold">Reports & Insights</CardTitle>
                <CardDescription>
                  Comprehensive analytics reports with interactive visualizations and actionable business insights
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300 hover:scale-105">
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-xl mb-4 flex items-center justify-center">
                  <Zap className="w-6 h-6 text-primary-foreground" />
                </div>
                <CardTitle className="font-serif font-bold">Real-time Processing</CardTitle>
                <CardDescription>
                  Live progress tracking with WebSocket connections for immediate feedback and status updates
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300 hover:scale-105">
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-secondary to-primary rounded-xl mb-4 flex items-center justify-center">
                  <Shield className="w-6 h-6 text-secondary-foreground" />
                </div>
                <CardTitle className="font-serif font-bold">Secure & Reliable</CardTitle>
                <CardDescription>
                  Enterprise-grade security with encrypted data processing and reliable cloud infrastructure
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300 hover:scale-105">
              <CardHeader>
                <div className="w-12 h-12 bg-gradient-to-br from-accent to-secondary rounded-xl mb-4 flex items-center justify-center">
                  <Users className="w-6 h-6 text-accent-foreground" />
                </div>
                <CardTitle className="font-serif font-bold">Team Collaboration</CardTitle>
                <CardDescription>
                  Share insights, collaborate on projects, and manage team access with advanced user management
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="bg-gradient-to-r from-primary/10 to-secondary/10 rounded-3xl p-12 border border-primary/20">
            <h2 className="font-serif font-black text-3xl sm:text-4xl mb-6 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Ready to Transform Your Data?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of data scientists and analysts who trust DataFlow Pro for their most important projects.
            </p>
            <Link href="/dashboard">
              <Button
                size="lg"
                className="bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90 text-lg px-8 py-6"
              >
                Start Your Journey
                <ArrowRight className="ml-2 w-5 h-5" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="font-serif font-black text-xl bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  DataFlow Pro
                </span>
              </div>
              <p className="text-muted-foreground mb-4 max-w-md">
                Professional AI-powered data science platform for modern teams and enterprises.
              </p>
            </div>

            <div>
              <h3 className="font-serif font-bold mb-4">Platform</h3>
              <ul className="space-y-2 text-muted-foreground">
                <li>
                  <Link href="/dashboard" className="hover:text-primary transition-colors">
                    Dashboard
                  </Link>
                </li>
                <li>
                  <Link href="/help" className="hover:text-primary transition-colors">
                    Help
                  </Link>
                </li>
                <li>
                  <Link href="/settings" className="hover:text-primary transition-colors">
                    Settings
                  </Link>
                </li>
                <li>
                  <Link href="/about" className="hover:text-primary transition-colors">
                    About
                  </Link>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-serif font-bold mb-4">Support</h3>
              <ul className="space-y-2 text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    Documentation
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    API Reference
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    Contact
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    Status
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-border/50 mt-8 pt-8 text-center text-muted-foreground">
            <p>&copy; 2024 DataFlow Pro. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
