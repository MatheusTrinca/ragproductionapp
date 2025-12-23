from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

load_dotenv()

app = FastAPI()


class IngestPDFRequest(BaseModel):
    pdf_path: str
    source_id: str | None = None


@app.post("/ingest_pdf")
async def ingest_pdf(request: IngestPDFRequest):
    pdf_path = request.pdf_path
    source_id = request.source_id or pdf_path

    # Load and chunk PDF
    chunks = load_and_chunk_pdf(pdf_path)

    # Embed and upsert to vector database
    vectors = embed_text(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}-{i}")) for i in range(len(chunks))]
    payloads = [{'source_id': source_id, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vectors, payloads)

    return {"ingested": source_id}
