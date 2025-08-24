"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  ArrowLeft,
  Sparkles,
  Target,
  Users,
  Zap,
  Shield,
  Globe,
  Heart,
  Code,
  Database,
  MessageSquare,
  BarChart3,
} from "lucide-react"
import Link from "next/link"

export function AboutPage() {
  const features = [
    {
      icon: Database,
      title: "Smart Data Processing",
      description: "AI-powered preprocessing with GPT-5 integration for intelligent data cleaning and transformation",
    },
    {
      icon: MessageSquare,
      title: "Conversational AI",
      description: "Interactive chatbot assistant for data analysis guidance and insights discovery",
    },
    {
      icon: BarChart3,
      title: "Advanced Analytics",
      description: "Comprehensive reporting with interactive visualizations and actionable business insights",
    },
    {
      icon: Zap,
      title: "Real-time Processing",
      description: "Live progress tracking with WebSocket connections for immediate feedback",
    },
    {
      icon: Shield,
      title: "Enterprise Security",
      description: "Bank-grade encryption and security measures to protect your sensitive data",
    },
    {
      icon: Globe,
      title: "Cloud-Native",
      description: "Scalable cloud infrastructure built for performance and reliability",
    },
  ]

  const team = [
    {
      name: "AI Research Team",
      role: "Machine Learning & NLP",
      description: "Developing cutting-edge AI algorithms for data analysis and natural language processing",
    },
    {
      name: "Data Engineering Team",
      role: "Infrastructure & Processing",
      description: "Building scalable data pipelines and processing infrastructure for enterprise workloads",
    },
    {
      name: "Product Team",
      role: "User Experience & Design",
      description: "Creating intuitive interfaces and seamless user experiences for data professionals",
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-card to-background">
      {/* Navigation */}
      <nav className="border-b border-border/50 backdrop-blur-sm bg-background/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-16">
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
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <Badge
            variant="secondary"
            className="mb-6 bg-gradient-to-r from-primary/10 to-secondary/10 text-primary border-primary/20"
          >
            <Heart className="w-4 h-4 mr-2" />
            Built with Passion for Data Science
          </Badge>

          <h1 className="font-serif font-black text-3xl sm:text-5xl mb-6 bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent leading-tight">
            Empowering Data Scientists
            <br />
            Worldwide
          </h1>

          <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
            DataFlow Pro is a next-generation AI-powered data science platform designed to democratize advanced
            analytics and make data insights accessible to everyone, from beginners to experts.
          </p>
        </div>

        {/* Mission & Vision */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-br from-primary to-secondary rounded-xl mb-4 flex items-center justify-center">
                <Target className="w-6 h-6 text-primary-foreground" />
              </div>
              <CardTitle className="font-serif font-bold text-2xl">Our Mission</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                To democratize data science by providing powerful, AI-driven tools that transform complex data analysis
                into intuitive, actionable insights. We believe every organization should have access to
                enterprise-grade data science capabilities.
              </p>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <div className="w-12 h-12 bg-gradient-to-br from-accent to-primary rounded-xl mb-4 flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-accent-foreground" />
              </div>
              <CardTitle className="font-serif font-bold text-2xl">Our Vision</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                A world where data-driven decision making is accessible to everyone. We envision a future where AI
                assistants work alongside humans to unlock the full potential of data, driving innovation and growth
                across all industries.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Features Overview */}
        <div className="mb-16">
          <div className="text-center mb-12">
            <h2 className="font-serif font-black text-3xl mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Platform Features
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Comprehensive tools designed for modern data science workflows
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const IconComponent = feature.icon
              return (
                <Card
                  key={index}
                  className="border-border/50 bg-card/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300"
                >
                  <CardHeader className="pb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg mb-3 flex items-center justify-center">
                      <IconComponent className="w-5 h-5 text-primary-foreground" />
                    </div>
                    <CardTitle className="font-serif font-bold">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground text-sm leading-relaxed">{feature.description}</p>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>

        {/* Technology Stack */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm mb-16">
          <CardHeader className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-secondary to-accent rounded-2xl mx-auto mb-4 flex items-center justify-center">
              <Code className="w-8 h-8 text-secondary-foreground" />
            </div>
            <CardTitle className="font-serif font-black text-2xl">Built with Modern Technology</CardTitle>
            <CardDescription className="text-lg">
              Leveraging cutting-edge tools and frameworks for optimal performance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
              <div>
                <h3 className="font-serif font-bold mb-2">Frontend</h3>
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>Next.js 15</p>
                  <p>React 18</p>
                  <p>TypeScript</p>
                  <p>Tailwind CSS</p>
                </div>
              </div>
              <div>
                <h3 className="font-serif font-bold mb-2">Backend</h3>
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>FastAPI</p>
                  <p>Python</p>
                  <p>WebSockets</p>
                  <p>Async Processing</p>
                </div>
              </div>
              <div>
                <h3 className="font-serif font-bold mb-2">AI & ML</h3>
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>GPT-5 Integration</p>
                  <p>Pandas</p>
                  <p>NumPy</p>
                  <p>Scikit-learn</p>
                </div>
              </div>
              <div>
                <h3 className="font-serif font-bold mb-2">Infrastructure</h3>
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>Cloud Native</p>
                  <p>Docker</p>
                  <p>Microservices</p>
                  <p>Auto Scaling</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Team */}
        <div className="mb-16">
          <div className="text-center mb-12">
            <h2 className="font-serif font-black text-3xl mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Our Team
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Passionate experts dedicated to advancing data science
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <Card key={index} className="border-border/50 bg-card/50 backdrop-blur-sm text-center">
                <CardHeader>
                  <div className="w-16 h-16 bg-gradient-to-br from-primary to-secondary rounded-full mx-auto mb-4 flex items-center justify-center">
                    <Users className="w-8 h-8 text-primary-foreground" />
                  </div>
                  <CardTitle className="font-serif font-bold">{member.name}</CardTitle>
                  <CardDescription className="text-primary font-medium">{member.role}</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground text-sm leading-relaxed">{member.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* CTA Section */}
        <Card className="border-border/50 bg-gradient-to-r from-primary/10 to-secondary/10 border-primary/20">
          <CardContent className="text-center py-12">
            <h2 className="font-serif font-black text-3xl mb-6 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Ready to Transform Your Data?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of data professionals who trust DataFlow Pro for their most important projects.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/dashboard">
                <Button
                  size="lg"
                  className="bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90"
                >
                  Get Started Now
                </Button>
              </Link>
              <Link href="/help">
                <Button variant="outline" size="lg" className="border-primary/20 hover:bg-primary/5 bg-transparent">
                  Learn More
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
