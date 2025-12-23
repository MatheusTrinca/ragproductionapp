import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    src: str = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: str


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int 