"use client"
import { useState, useRef, useEffect } from "react"
import type React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Avatar } from "@/components/ui/avatar"
import { Loader2 } from "lucide-react"

// 메시지 타입 정의
export type Message = {
  id: string;
  role: "user" | "assistant";
  content?: string; 
}

export default function ChatbotInterface() {
  // 상태 관리
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  console.log({ messages });

  // 메시지 목록을 자동 스크롤하기 위한 ref
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 새 메시지가 추가될 때마다 스크롤 아래로 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // 메시지 전송 처리
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim()) return

    // 사용자 메시지 추가
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // API 호출
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...messages, userMessage],
        }),
      })

      if (!response.ok) {
        throw new Error("API 응답 오류")
      }

      const data = await response.json()
      console.log({ data });

      // 응답 메시지 추가
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.messages[0].content,
      }
      if (data.messages[0].content !== "") {
        setMessages((prev) => [...prev, assistantMessage])
      } else {
        console.log("백엔드 응답이 없습니다.")
      }
    } catch (error) {
      console.error("메시지 전송 오류:", error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <h1 className="text-xl font-semibold">AI 챗봇</h1>
        </div>
      </header>

      <main className="flex-1 container py-4 md:py-8">
        <Card className="w-full max-w-3xl mx-auto border rounded-lg shadow-sm">
          <div className="flex flex-col h-[70vh]">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center h-full text-center text-muted-foreground">
                  <div>
                    <p className="mb-2">AI 챗봇에 오신 것을 환영합니다!</p>
                    <p>무엇이든 물어보세요.</p>
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
              <div ref={messagesEndRef} />
            </div>

            <div className="border-t p-4">
              <form onSubmit={handleSubmit} className="flex gap-2">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="메시지를 입력하세요..."
                  className="flex-1"
                  disabled={isLoading}
                />
                <Button type="submit" disabled={isLoading || !input.trim()}>
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "전송"}
                </Button>
              </form>
            </div>
          </div>
        </Card>
      </main>
    </div>
  )
}
