import os

from fastapi import FastAPI
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda


# ==============================
# Get Gemini API Key
# ==============================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


# ==============================
# Load KI Document
# ==============================

with open("AI_and_Education_KI.txt", "r", encoding="utf-8") as file:
    knowledge = file.read()


# ==============================
# Split KI into Chunks
# ==============================

chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(knowledge):
    end = start + chunk_size

    chunks.append(knowledge[start:end])

    start = end - overlap


# ==============================
# Gemini Model
# ==============================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ==============================
# RAG Function
# ==============================

def rag_answer(question):

    question_words = question.lower().split()

    best_chunk = chunks[0]
    best_score = 0

    for chunk in chunks:

        chunk_lower = chunk.lower()

        score = sum(
            1
            for word in question_words
            if word in chunk_lower
        )

        if score > best_score:
            best_score = score
            best_chunk = chunk


    prompt = f"""
You are an AI Education RAG assistant.

Answer the user's question using ONLY the information
provided in the KI below.

If the answer is not present in the KI, say:

"I could not find this information in the provided KI."

Do not make up information.

KI:
{best_chunk}

Question:
{question}

Give a short, clear and accurate answer.
"""


    response = llm.invoke(prompt)

    return response.content


# ==============================
# LangServe Chain
# ==============================

rag_chain = RunnableLambda(rag_answer)


# ==============================
# FastAPI Application
# ==============================

app = FastAPI(
    title="AI Education RAG System",
    version="1.0",
    description="RAG system based on Artificial Intelligence and Education KI."
)


# ==============================
# Home Route
# ==============================

@app.get("/")
def root():

    return {
        "message": "AI Education RAG System is running.",
        "playground": "/rag/playground/"
    }


# ==============================
# LangServe Route
# ==============================

add_routes(
    app,
    rag_chain,
    path="/rag"
)


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )