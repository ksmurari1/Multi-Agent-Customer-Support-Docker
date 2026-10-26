# Testing Guide

## 1. Pre-flight

1. Activate the intended Python environment.
2. Install dependencies from `requirements.txt`.
3. Keep credentials in `.env`; do not paste them into logs or screenshots.
4. Run `python -m pip check`.
5. Run `python test_e2e.py` for the offline test suite.

## 2. Functional test matrix

| ID | Scenario | Example | Expected behavior |
|---|---|---|---|
| T01 | Sample document indexing | Click **Load samples** | Sample files are listed and indexed without an extraction error |
| T02 | KB answer found | “How do I reset my password?” | Knowledge Assistant answers from a sample document; source file/location is shown; web task is skipped |
| T03 | KB answer found in another file | Ask about billing/shipping policy | Answer is grounded in the matching billing/shipping document |
| T04 | Web fallback | “What is the latest stable Python version?” | If the sample docs do not answer it, web task runs and shows URLs returned by the search provider |
| T05 | No documents | Clear the KB and ask a question | Web search route runs directly |
| T06 | Unsupported file | Upload a file type outside PDF/TXT/DOCX | UI rejects it or reports a clear supported-format message |
| T07 | Empty question | Submit blank input | Validation prevents a crew run |
| T08 | Answer length | Ask a normal question | Final answer and sources fit within the ten-line cap |
| T09 | Logging | Ask any question | `answers.txt` gains a new entry with query, route/evidence, answer, sources and timing |
| T10 | Search unavailable | Temporarily remove/disable search credentials or network | App reports search failure/fallback honestly; it must not invent URLs |
| T11 | Agent indicators | Run both a KB answer and a web fallback | KB route shows web skipped when sufficient; web route shows web active/done when invoked; Entry Agent shows saved |
| T12 | Secret hygiene | Inspect Git status | `.env`, `answers.txt`, caches and virtual environments are not staged |

## 3. Manual smoke test (recommended demo)

1. Click **Test connections** and note which providers are available.
2. Load sample documents.
3. Ask `How do I reset my password?`. Check the source points to the appropriate account guide.
4. Ask `What is the latest stable Python version?`. Confirm that the web search indicator becomes active only when the KB answer is insufficient.
5. Inspect the saved record and download `answers.txt`.
6. Clear documents and ask another question to verify direct-to-web behavior.

## 4. Automated tests

Run from the project directory with the project environment activated:

```cmd
python test_e2e.py
```

Review the test file for supported options before running live integration tests. Tests that use live API keys may consume API credits or rely on network availability. Record the exact command and outcome in your buildathon notes.

## 5. Acceptance criteria

- [ ] Sample and user-uploaded documents can be indexed.
- [ ] A question answerable from a document does not trigger web search.
- [ ] An unanswerable question triggers web search when the provider is available.
- [ ] Sources refer only to retrieved document locations or URLs returned by search.
- [ ] The answer stays within the configured line limit.
- [ ] Agent statuses distinguish completed, skipped and failed stages accurately.
- [ ] Every attempted interaction is recorded, including an error where logging is possible.
- [ ] No secrets, logs or private documents are committed.
