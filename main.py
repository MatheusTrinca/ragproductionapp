from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import uuid
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

load_dotenv()

client = OpenAI()
app = FastAPI()


class IngestPDFRequest(BaseModel):
    pdf_path: str
    source_id: str | None = None

class QuestionRequest(BaseModel):
    question: str


@app.post("/ingest_pdf")
async def ingest_pdf(request: IngestPDFRequest) -> RAGUpsertResult:
    pdf_path = request.pdf_path
    source_id = request.source_id or pdf_path

    # Load and chunk PDF
    chunks = load_and_chunk_pdf(pdf_path)

    # Embed and upsert to vector database
    vectors = embed_text(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}-{i}")) for i in range(len(chunks))]
    payloads = [{'source_id': source_id, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vectors, payloads)

    return RAGUpsertResult(ingested=source_id)

@app.post("/query")
async def query(request: QuestionRequest) -> RAGQueryResult:
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vector = embed_text([question])[0]
        results = QdrantStorage().search(query_vector, top_k)
        return RAGSearchResult(contexts=results['contexts'], sources=results['sources'])

    found = _search(request.question)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {request.question}\n"
        "Answer concisely using the context above."
    )

    # Generate answer using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": user_content}
        ]
    )

    answer = response.choices[0].message.content

    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    )  
