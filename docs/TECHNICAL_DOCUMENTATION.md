# Technical Documentation — Multi-Agent Customer Support

## 1. Purpose

This application answers customer-support questions from user-supplied documents first and uses web search only when document evidence is judged insufficient. It provides a Streamlit UI, a CrewAI sequential workflow and an append-only local interaction log.

## 2. System components

| Component | Responsibility | Implementation notes |
|---|---|---|
| Streamlit UI | Upload/load documents, accept a query, show agent status and render answer/source tabs | `app.py` |
| Document loader | Extract text from PDF, DOCX and TXT; retain page/section labels when available | `extract_pages()` |
| Chunker | Split text into chunks with overlap | Defaults: 900 characters, 150-character overlap |
| Embedder | Produce vectors for semantic retrieval | OpenAI `text-embedding-3-small` when selected; local hashing embedder option |
| Dense retrieval | Find semantically similar chunks | FAISS |
| Sparse retrieval | Find keyword matches | BM25 (`rank-bm25`) |
| Fusion/reranking | Combine dense and sparse ranks and rank candidate passages | Reciprocal Rank Fusion (RRF) plus app-level reranking |
| Knowledge Assistant | Answer strictly from retrieved passages and state whether they are sufficient | CrewAI agent/task |
| Web Search Assistant | Search web results and provide URLs if KB evidence is insufficient | Conditional CrewAI task; Serper with DuckDuckGo fallback |
| Entry Agent | Record interaction details without asking an LLM to reconstruct the record | Recorder wrapper calls local logging action |
| Interaction log | Preserve query, evidence, answer, sources and timing | `answers.txt`, append-only |

## 3. Runtime data flow

### Document indexing

1. The user loads bundled sample files or uploads PDF/TXT/DOCX files.
2. Text is extracted while preserving document names and page/section labels where supported.
3. Text is divided into overlapping chunks.
4. Embeddings are created according to `EMBEDDINGS_BACKEND`.
5. FAISS dense search and BM25 keyword search rank chunks; RRF/reranking produces the final evidence set.

### Question answering

1. Streamlit receives the user's question.
2. If a knowledge base exists, the Knowledge Assistant receives the top retrieved passages and answers only from them.
3. The task evaluates evidence sufficiency. If sufficient, the web task is skipped. If insufficient, the conditional Web Search Assistant searches the web. If no knowledge base is loaded, the app routes directly to web search.
4. The final answer is assembled with only known document references or URLs returned by the search results. Output is constrained to approximately ten lines, including sources.
5. The Entry Agent triggers the local recorder to append query, retrieval findings, final response, sources, route and timing information to `answers.txt`. A safety path attempts to log the interaction if the crew errors.
6. UI indicators reflect active, done, skipped or error states emitted by the pipeline.

## 4. CrewAI orchestration

The application uses `Process.sequential`. When documents are loaded, the logical task sequence is Knowledge Assistant → conditional Web Search Assistant → Entry Agent. The web task's condition reads the Knowledge Assistant's structured output. With no loaded documents, the KB task is omitted and web search runs before the Entry Agent.

Although the UI calls these “agents,” the Entry Agent's LLM is wrapped so the actual record write is deterministic local code. This avoids asking a model to copy structured answers back into a log and reduces latency and transcription errors.

## 5. Configuration

| Variable | Required? | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Required for model answers | OpenAI authentication |
| `SERPER_API_KEY` | Optional | Primary web search provider; app may fall back to DuckDuckGo |
| `OPENAI_MODEL` | Optional | Chat model; default `gpt-4o-mini` |
| `OPENAI_EMBED_MODEL` | Optional | Embedding model; default `text-embedding-3-small` |
| `EMBEDDINGS_BACKEND` | Optional | `auto`, `openai` where supported, or `local`; `local` forces local hashing embeddings |

`.env` is loaded from the application directory. Never commit `.env`.

## 6. Key defaults

- Chunk size: 900 characters
- Chunk overlap: 150 characters
- Retrieval top-k: 5
- Maximum indexed chunks: 4,000
- Maximum upload size: 25 MB per file in the app configuration
- Maximum final answer: 10 lines including source lines
- Maximum displayed source count: 3

These are implementation defaults, not universal best-practice values. Tune only after evaluating representative questions against the intended document set.

## 7. Observability and persistence

The UI pipeline rail is updated through emitted events. `RunState` tracks retrieval hits, web results, selected route, per-stage timing, errors and whether the record was saved. `answers.txt` is local append-only output and can be downloaded through the UI. It is not a central database and does not provide multi-user concurrency guarantees beyond the in-process write lock.

## 8. Error handling and known limitations

- Missing/invalid search credentials may result in fallback to DuckDuckGo, subject to network/provider availability.
- Scanned PDFs without extractable text are not OCR-processed.
- Local hashing embeddings are a lightweight fallback, not equivalent to a trained semantic embedding model.
- LLM judgments of evidence sufficiency can be imperfect; validate with positive and negative retrieval tests.
- This app is a local buildathon/demo, not a production-secured service. It has no built-in user authentication, authorization or centralized audit database.

## 9. Security guidance

- Keep API keys in `.env` and rotate any key accidentally exposed.
- Do not commit `.env`, `answers.txt`, user uploads or local caches.
- Review documents and logs for confidential customer information before sharing.
- Treat uploaded document contents as untrusted data; the Knowledge Assistant prompt explicitly tells the model not to follow instructions embedded in passages.

## 10. Repository entry points

- `app.py`: application, ingestion, retrieval, CrewAI tasks and Streamlit rendering
- `test_e2e.py`: regression tests
- `requirements.txt`: dependencies
- `sample_docs/`: sample knowledge base
- `architecture.png`: rendered architecture
- `docs/architecture.mmd.md`: editable Mermaid diagram source
