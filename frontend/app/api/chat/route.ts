import { NextResponse } from "next/server"

export async function POST(req: Request) {
  type Message = {
    id: string;
    role: 'user' | 'assistant';
    content?: string; // 일부 메시지는 content가 없으므로 optional
  };

  try {
    const { messages }: { messages: Message[] } = await req.json();

    const filteredMessages = messages
      .filter((msg: Message) => msg.content !== undefined)
      .filter((msg: Message) => msg.role === 'user')
      .map((msg: Message) => ({
        role: msg.role,
        content: msg.content as string,
      }));

    const response = await fetch('http://KiChatserver:8000/chat', {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages: filteredMessages }),
    });


    const data = await response.json();
    console.log({ data });

    return NextResponse.json({ messages: [data] })
  } catch (error) {
    console.error("채팅 API 오류:", error)
    return new Response("서버 오류가 발생했습니다", { status: 500 })
  }
}