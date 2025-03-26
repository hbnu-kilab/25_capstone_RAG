"use client"
import { useChat } from "@ai-sdk/react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Avatar } from "@/components/ui/avatar"
import { Loader2 } from "lucide-react"

export default function ChatbotInterface() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: "/api/chat", // This would connect to your backend when implemented
  })

  return (
    <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <h1 className="text-xl font-semibold">AI Chatbot</h1>
        </div>
      </header>

      <main className="flex-1 container py-4 md:py-8">
        <Card className="w-full max-w-3xl mx-auto border rounded-lg shadow-sm">
          <div className="flex flex-col h-[70vh]">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center h-full text-center text-muted-foreground">
                  <div>
                    <p className="mb-2">Welcome to the AI Chatbot!</p>
                    <p>Ask me anything to get started.</p>
                  </div>
                </div>
              ) : (
                messages.map((message) => (
                  <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`flex items-start gap-2 max-w-[80%] ${message.role === "user" ? "flex-row-reverse" : ""}`}
                    >
                      <Avatar className="mt-1">
                        <div
                          className={`flex h-full w-full items-center justify-center rounded-full ${
                            message.role === "user" ? "bg-primary" : "bg-secondary"
                          }`}
                        >
                          {message.role === "user" ? "U" : "AI"}
                        </div>
                      </Avatar>
                      <div
                        className={`rounded-lg px-4 py-2 ${
                          message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                        }`}
                      >
                        {message.content}
                      </div>
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-start gap-2 max-w-[80%]">
                    <Avatar className="mt-1">
                      <div className="flex h-full w-full items-center justify-center rounded-full bg-secondary">AI</div>
                    </Avatar>
                    <div className="rounded-lg px-4 py-2 bg-muted">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="border-t p-4">
              <form onSubmit={handleSubmit} className="flex gap-2">
                <Input
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Type your message..."
                  className="flex-1"
                  disabled={isLoading}
                />
                <Button type="submit" disabled={isLoading || !input.trim()}>
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Send"}
                </Button>
              </form>
            </div>
          </div>
        </Card>
      </main>
    </div>
  )
}

