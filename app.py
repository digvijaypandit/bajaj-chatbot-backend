from fastapi import FastAPI
from pydantic import BaseModel
from gemini_client import ask_gemini
from stock_query import get_stats_for_month, compare_months
from rag_qa import answer_question_with_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bajaj-chatbot-frontend.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatInput(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Bajaj Finserv GenAI Bot is running!"}

@app.post("/chatbot")
def chatbot(input: ChatInput):
    question = input.question.strip()

    if "compare" in question.lower() and "and" in question.lower():
        try:
            # Extract months from question
            parts = question.lower().split("compare")[1].split("and")
            from_month = parts[0].strip().title()
            to_month = parts[1].strip().title()
            return compare_months(from_month, to_month)
        except:
            return {"error": "Please ask like: Compare Jan-2024 and Apr-2024"}

    elif "stock" in question.lower() or "price" in question.lower():
        for word in question.split():
            if "-" in word:
                return get_stats_for_month(word.title())

    else:
        return {"response": answer_question_with_rag(question)}
