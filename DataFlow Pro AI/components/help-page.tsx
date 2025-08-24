"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import {
  ArrowLeft,
  HelpCircle,
  Search,
  Book,
  MessageCircle,
  Mail,
  Sparkles,
  Database,
  MessageSquare,
  BarChart3,
} from "lucide-react"
import Link from "next/link"

export function HelpPage() {
  const [searchQuery, setSearchQuery] = useState("")

  const faqs = [
    {
      question: "How do I upload a dataset for analysis?",
      answer:
        "Navigate to the Data Science Agent module from your dashboard. Click 'Choose File' to select your CSV or Excel file (up to 100MB). Optionally enter your GPT-5 API key for enhanced analysis, then click 'Start Analysis' to begin processing.",
      category: "Data Science Agent",
    },
    {
      question: "What file formats are supported?",
      answer:
        "Currently, we support CSV (.csv) and Excel files (.xlsx, .xls). Files should be properly formatted with headers in the first row. Maximum file size is 100MB.",
      category: "Data Science Agent",
    },
    {
      question: "How long does data processing take?",
      answer:
        "Processing time varies based on dataset size and complexity. Small datasets (< 1MB) typically process in 1-2 minutes, while larger datasets may take 5-15 minutes. You'll see real-time progress updates during processing.",
      category: "Data Science Agent",
    },
    {
      question: "Can I use my own GPT-5 API key?",
      answer:
        "Yes! You can enter your own GPT-5 API key during upload or save it in Settings. This ensures you have full control over API usage and costs. If no key is provided, we'll use our default key with usage limits.",
      category: "API & Settings",
    },
    {
      question: "What happens to my data after processing?",
      answer:
        "Your data is processed securely and stored temporarily for result access. By default, all files are automatically deleted after 30 days. You can modify this in Privacy & Security settings.",
      category: "Privacy & Security",
    },
    {
      question: "How do I download my results?",
      answer:
        "Once processing is complete, you'll see download buttons for: Processed Dataset (cleaned data), Analysis JSON (detailed insights), HTML Report (visual summary), and Processing Logs. Click any button to download the respective file.",
      category: "Results & Downloads",
    },
    {
      question: "When will the AI Chatbot be available?",
      answer:
        "The AI Chatbot module is currently in development and will be available in the next major update. It will provide conversational data analysis guidance and insights based on your processed datasets.",
      category: "Upcoming Features",
    },
    {
      question: "Can I collaborate with team members?",
      answer:
        "Team collaboration features are planned for a future release. This will include shared workspaces, result sharing, and user management capabilities.",
      category: "Upcoming Features",
    },
  ]

  const guides = [
    {
      title: "Getting Started with Data Science Agent",
      description: "Complete walkthrough of uploading and analyzing your first dataset",
      icon: Database,
      steps: ["Upload your CSV/Excel file", "Configure API settings", "Monitor real-time progress", "Download results"],
    },
    {
      title: "Understanding Your Analysis Results",
      description: "Learn how to interpret the AI-generated insights and recommendations",
      icon: BarChart3,
      steps: [
        "Review data quality score",
        "Understand preprocessing steps",
        "Interpret recommendations",
        "Export findings",
      ],
    },
    {
      title: "API Key Management",
      description: "How to configure and manage your GPT-5 API keys securely",
      icon: MessageSquare,
      steps: ["Obtain GPT-5 API key", "Add key to settings", "Monitor usage", "Update as needed"],
    },
  ]

  const filteredFaqs = faqs.filter(
    (faq) =>
      faq.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.answer.toLowerCase().includes(searchQuery.toLowerCase()) ||
      faq.category.toLowerCase().includes(searchQuery.toLowerCase()),
  )

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
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="font-serif font-black text-3xl sm:text-4xl mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
            Help Center
          </h1>
          <p className="text-xl text-muted-foreground mb-8">Find answers, guides, and support for DataFlow Pro</p>

          {/* Search */}
          <div className="max-w-md mx-auto relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search help articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Quick Guides */}
          <div className="lg:col-span-1">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm mb-8">
              <CardHeader>
                <CardTitle className="font-serif font-bold flex items-center">
                  <Book className="w-5 h-5 mr-2" />
                  Quick Guides
                </CardTitle>
                <CardDescription>Step-by-step tutorials to get you started</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {guides.map((guide, index) => {
                  const IconComponent = guide.icon
                  return (
                    <div
                      key={index}
                      className="p-4 border border-border/50 rounded-lg hover:bg-muted/50 transition-colors cursor-pointer"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center flex-shrink-0">
                          <IconComponent className="w-4 h-4 text-primary-foreground" />
                        </div>
                        <div className="flex-1">
                          <h3 className="font-serif font-bold text-sm mb-1">{guide.title}</h3>
                          <p className="text-xs text-muted-foreground mb-2">{guide.description}</p>
                          <div className="space-y-1">
                            {guide.steps.map((step, stepIndex) => (
                              <div key={stepIndex} className="flex items-center text-xs text-muted-foreground">
                                <div className="w-1 h-1 bg-primary rounded-full mr-2" />
                                {step}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </CardContent>
            </Card>

            {/* Contact Support */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="font-serif font-bold flex items-center">
                  <MessageCircle className="w-5 h-5 mr-2" />
                  Need More Help?
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button variant="outline" className="w-full justify-start bg-transparent">
                  <Mail className="w-4 h-4 mr-2" />
                  Contact Support
                </Button>
                <Button variant="outline" className="w-full justify-start bg-transparent">
                  <MessageCircle className="w-4 h-4 mr-2" />
                  Live Chat
                </Button>
                <div className="text-center pt-4 border-t border-border/50">
                  <p className="text-sm text-muted-foreground">
                    Average response time: <span className="font-medium">2 hours</span>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* FAQ Section */}
          <div className="lg:col-span-2">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="font-serif font-bold flex items-center">
                  <HelpCircle className="w-5 h-5 mr-2" />
                  Frequently Asked Questions
                </CardTitle>
                <CardDescription>
                  {searchQuery ? `${filteredFaqs.length} results found` : "Common questions and answers"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="space-y-4">
                  {filteredFaqs.map((faq, index) => (
                    <AccordionItem
                      key={index}
                      value={`item-${index}`}
                      className="border border-border/50 rounded-lg px-4"
                    >
                      <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-start space-x-3 text-left">
                          <Badge variant="outline" className="text-xs mt-1 flex-shrink-0">
                            {faq.category}
                          </Badge>
                          <span className="font-medium">{faq.question}</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="text-muted-foreground pt-2 pb-4">{faq.answer}</AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>

                {filteredFaqs.length === 0 && searchQuery && (
                  <div className="text-center py-8 text-muted-foreground">
                    <HelpCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No results found for "{searchQuery}"</p>
                    <p className="text-sm mt-2">Try different keywords or contact support</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
