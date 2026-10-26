"""Offline end-to-end tests for app.py (no API keys, no internet needed).

Real components under test : CrewAI (sequential crew + ConditionalTask), FAISS, BM25, pypdf, python-docx,
                              the Streamlit UI (via streamlit.testing AppTest), the answers.txt log.
Faked components           : the chat LLM (FakeLLM below) and the web-search backend (canned results).

Run:   python test_e2e.py            (or:  pytest -q test_e2e.py)
Live smoke test with real keys:  python test_e2e.py --live
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["EMBEDDINGS_BACKEND"] = "local"            # no OpenAI embedding calls in offline tests
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app  # noqa: E402

SAMPLES = sorted((HERE / "sample_docs").glob("*"))
FILES = [(p.name, p.read_bytes()) for p in SAMPLES if p.suffix.lower() in (".pdf", ".docx", ".txt")]


# ----------------------------------------------------------------------------- fakes
def make_fake_llm_class(hallucinate_url: bool = False):
    from crewai.llms.base_llm import BaseLLM

    class FakeLLM(BaseLLM):
        llm_type: str = "fake"
        calls: int = 0

        def supports_function_calling(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            self.calls += 1
            text = messages if isinstance(messages, str) else "\n".join(str(m.get("content", "")) for m in messages)
            if "PASSAGES:" in text:                                          # Knowledge Assistant
                q = re.search(r"QUESTION: (.*)", text).group(1)
                qt = set(app.tokenize(q))
                best, best_n = None, 0
                for m in re.finditer(r"\[(D\d+)\] [^\n]*\n(.*?)(?=\n\n\[D\d+\]|\Z)", text.split("PASSAGES:")[1], re.S):
                    n = len(qt & set(app.tokenize(m.group(2))))
                    if n > best_n:
                        best, best_n = m, n
                if best is not None and best_n >= 2:
                    ans = "\n".join(best.group(2).strip().split("\n")[:3])
                    import json
                    return "Thought: I now know the final answer\nFinal Answer: " + json.dumps(
                        {"sufficient": True, "answer": ans, "sources": [best.group(1)], "missing": ""})
                return 'Thought: I now know the final answer\nFinal Answer: {"sufficient": false, "answer": "", "sources": [], "missing": "not covered"}'
            if "web_search" in text and "Customer question" in text:        # Web Search Assistant (ReAct)
                if "URL: http" not in text:             # no tool observation yet
                    q = re.search(r"Customer question: (.*)", text).group(1)
                    return f'Thought: I should search.\nAction: web_search\nAction Input: {{"query": "{q}"}}'
                urls = re.findall(r"URL: (\S+)", text)
                import json
                used = urls[:1] + (["https://made-up.example/hallucinated"] if hallucinate_url else [])
                return "Thought: I now know the final answer\nFinal Answer: " + json.dumps(
                    {"answer": "Python 3.14 is the latest stable release.\n" + "\n".join(f"extra line {i}" for i in range(14)),
                     "urls": used})
            return "Thought: done\nFinal Answer: ok"

    return FakeLLM


def fake_search(query):
    return ([{"title": "Python Releases", "snippet": "Latest stable version is 3.14", "url": "https://www.python.org/downloads/"},
             {"title": "What's new", "snippet": "Release notes", "url": "https://docs.python.org/3/whatsnew/"}], "FakeSearch", [])


def empty_search(query):
    return ([], "none", ["Serper failed: HTTP 403 (key rejected / no credits)", "DuckDuckGo failed: Timeout"])


def run(query, kb, search=fake_search, hallucinate=False):
    events = []
    app.LOG_FILE = Path(tempfile.mkdtemp()) / "answers.txt"
    cls = make_fake_llm_class(hallucinate)
    state = app.run_pipeline(query, kb, lambda a, s, n, t: events.append((a, s, n)),
                             llm_factory=lambda: cls(model="fake"), search_fn=search)
    return state, events


def statuses(events):
    out = {}
    for a, s, _ in events:
        out.setdefault(a, []).append(s)
    return out


# ----------------------------------------------------------------------------- tests
def test_loaders_and_chunking():
    seen = {}
    for name, data in FILES:
        pages = app.extract_pages(name, data)
        chunks = [c for _, t in pages for c in app.split_text(t)]
        assert chunks, f"no chunks from {name}"
        seen[name] = (len(pages), len(chunks))
    assert seen["Nimbus_Account_Guide.pdf"][0] == 2           # PDF keeps page numbers
    assert any(loc.startswith("§") for loc, _ in app.extract_pages("Nimbus_Refund_Policy.docx", dict(FILES)["Nimbus_Refund_Policy.docx"]))
    long_text = ("Sentence number one is here. " * 200)
    assert all(len(c) <= app.CHUNK_CHARS + app.CHUNK_OVERLAP + 50 for c in app.split_text(long_text))


def test_kb_hybrid_retrieval():
    kb, rep = app.build_knowledge_base(FILES)
    assert kb is not None and kb.size >= 8 and not rep.warnings
    import faiss, rank_bm25                                  # noqa: F401
    assert kb.index.ntotal == kb.size
    cases = {"How do I reset my password?": "Nimbus_Account_Guide.pdf",
             "refund annual plan within 30 days": "Nimbus_Refund_Policy.docx",
             "express delivery cost hardware keys": "Nimbus_Billing_and_Shipping.pdf",
             "support hours live chat": "Nimbus_FAQ.txt"}
    for q, doc in cases.items():
        hits = kb.search(q)
        assert hits[0].chunk.doc == doc, (q, [h.ref for h in hits])
    assert kb.search("password")[0].chunk.loc == "p.1"


def test_duplicate_uploads_are_indexed_once():
    base, _ = app.build_knowledge_base(FILES)
    faq = dict(FILES)["Nimbus_FAQ.txt"]
    kb, rep = app.build_knowledge_base(FILES + [("Nimbus_FAQ.txt", faq), ("renamed_copy.txt", faq)])   # same name AND renamed copy
    assert kb.size == base.size, (kb.size, base.size)
    assert kb.index.ntotal == base.size and len(kb.bm25.corpus_size.__class__.__name__) > 0
    assert sum(1 for f in rep.files if "duplicate" in f["note"]) == 2
    assert not any("scanned" in w for w in rep.warnings), rep.warnings
    assert [n for n, _ in kb.doc_summary()].count("Nimbus_FAQ.txt") == 1
    # same text, different bytes (e.g. trailing newline): caught at chunk level, not indexed twice
    txt = b"Refund policy: annual plans can be refunded within 30 days of purchase or renewal."
    kb2, rep2 = app.build_knowledge_base([("a.txt", txt), ("b.txt", txt + b"\n")])
    assert kb2.size == 1 and "already indexed" in rep2.files[1]["note"], (kb2.size, rep2.files)


def test_bad_inputs():
    kb, rep = app.build_knowledge_base([("empty.txt", b""), ("junk.pdf", b"not a pdf"), ("x.xyz", b"hi")])
    assert kb is None and len(rep.warnings) == 3
    kb, rep = app.build_knowledge_base(FILES + [("junk.pdf", b"not a pdf")])
    assert kb is not None and any("junk.pdf" in w for w in rep.warnings)


def test_pipeline_kb_answers_and_web_is_skipped():
    kb, _ = app.build_knowledge_base(FILES)
    state, ev = run("How do I reset my password?", kb)
    s = statuses(ev)
    assert s["kb"][-1] == "done" and s["web"] == ["skipped"] and s["entry"] == ["active", "done"], s
    assert not state.web_ran and state.final_origin == "documents"
    assert "Nimbus_Account_Guide.pdf" in state.final_sources[0]["label"]
    assert {n for n, _ in state.docs} == {n for n, _ in FILES}
    assert len(state.final_lines) + len(state.final_sources) <= app.MAX_LINES
    log = app.LOG_FILE.read_text(encoding="utf-8")
    assert "Query: How do I reset my password?" in log and "skipped (documents were sufficient)" in log and "Route: knowledge base" in log
    assert state.error == ""


def test_pipeline_falls_back_to_web():
    kb, _ = app.build_knowledge_base(FILES)
    state, ev = run("What is the latest stable Python version?", kb)
    s = statuses(ev)
    assert s["kb"][-1] == "done" and s["web"] == ["active", "done"] and s["entry"][-1] == "done", s
    assert state.web_ran and state.final_origin == "web"
    assert state.final_sources[0]["url"] == "https://www.python.org/downloads/"
    assert len(state.final_lines) + len(state.final_sources) <= app.MAX_LINES, "answer must be <= 10 lines"
    log = app.LOG_FILE.read_text(encoding="utf-8")
    assert "Engine: FakeSearch" in log and "Route: knowledge base + web" in log and "[D1]" in log


def test_hallucinated_urls_are_dropped():
    state, _ = run("What is the latest stable Python version?", None, hallucinate=True)
    assert all("made-up.example" not in s["url"] for s in state.final_sources)


def test_no_documents_goes_straight_to_web():
    state, ev = run("What is the latest stable Python version?", None)
    s = statuses(ev)
    assert s["kb"] == ["skipped"] and s["web"] == ["active", "done"] and s["entry"] == ["active", "done"], s
    assert state.route == "web only" and app.LOG_FILE.exists()


def test_web_failure_is_reported_honestly():
    kb, _ = app.build_knowledge_base(FILES)
    state, ev = run("What is the latest stable Python version?", kb, search=empty_search)
    assert state.final_origin == "none" and "HTTP 403" in "\n".join(state.final_lines)
    assert statuses(ev)["entry"][-1] == "done" and app.LOG_FILE.exists()


def test_web_search_fallback_chain():
    """Serper 403 -> DuckDuckGo fallback; both failing -> clear notes, no crash."""
    import types
    from unittest import mock

    class R403:
        status_code = 403
        def json(self):
            return {}

    class FakeDDGS:
        def __init__(self, **kw):
            pass
        def text(self, q, max_results=5):
            return [{"title": "DDG hit", "body": "snippet", "href": "https://example.org/a"}]

    os.environ["SERPER_API_KEY"] = "bad-key"
    with mock.patch("requests.post", lambda *a, **k: R403()), mock.patch.dict(sys.modules, {"ddgs": types.SimpleNamespace(DDGS=FakeDDGS)}):
        res, engine, notes = app.web_search("anything")
    assert engine == "DuckDuckGo" and res[0]["url"] == "https://example.org/a"
    assert any("HTTP 403" in n and "no credits" in n for n in notes), notes
    os.environ["SERPER_API_KEY"] = ""
    with mock.patch.dict(sys.modules, {"ddgs": types.SimpleNamespace(DDGS=type("B", (), {"__init__": lambda s, **k: None, "text": lambda s, q, max_results=5: (_ for _ in ()).throw(RuntimeError("x"))}))}):
        res, engine, notes = app.web_search("anything")
    assert res == [] and engine == "none" and any("not set" in n for n in notes)


def test_openai_wire_protocol_with_local_server():
    """Runs CrewAI's REAL OpenAI LLM client + the app's REAL OpenAI embedder against a local OpenAI-compatible
    server (native tool-calling included). Proves the live code path works up to the network boundary."""
    import json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {"embeddings": 0, "chat": 0, "tool_calls": 0, "modes": set()}
    hasher = app.HashingEmbedder()

    def chat_reply(body):
        msgs = body.get("messages", [])
        text = "\n".join(str(m.get("content", "")) for m in msgs)
        has_tools = bool(body.get("tools"))
        seen["modes"].add("native-tools" if has_tools else "react-text")
        if "PASSAGES:" in text:
            q = re.search(r"QUESTION: (.*)", text).group(1)
            qt = set(app.tokenize(q)); best = None; best_n = 0
            for m in re.finditer(r"\[(D\d+)\] [^\n]*\n(.*?)(?=\n\n\[D\d+\]|\Z)", text.split("PASSAGES:")[1], re.S):
                n = len(qt & set(app.tokenize(m.group(2))))
                if n > best_n:
                    best, best_n = m, n
            if best is not None and best_n >= 2:
                return json.dumps({"sufficient": True, "answer": best.group(2).strip().split("\n")[0], "sources": [best.group(1)], "missing": ""})
            return json.dumps({"sufficient": False, "answer": "", "sources": [], "missing": "x"})
        if "web_search" in text and "Customer question" in text:
            if any(m.get("role") == "tool" for m in msgs) or "URL: http" in text:
                url = re.findall(r"URL: (\S+)", text)[0]
                return json.dumps({"answer": "Python 3.14 is the latest stable release.", "urls": [url]})
            q = re.search(r"Customer question: (.*)", text).group(1)
            if has_tools:
                return {"tool": ("web_search", json.dumps({"query": q}))}
            return f'Thought: search.\nAction: web_search\nAction Input: {{"query": "{q}"}}'
        return "ok"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                seen["embeddings"] += 1
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vecs = hasher.embed(inputs).tolist()
                out = {"object": "list", "model": body["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1},
                       "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]}
            else:
                seen["chat"] += 1
                r = chat_reply(body)
                if isinstance(r, dict):
                    seen["tool_calls"] += 1
                    name, args = r["tool"]
                    msg = {"role": "assistant", "content": None,
                           "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}]}
                    fin = "tool_calls"
                else:
                    msg, fin = {"role": "assistant", "content": r}, "stop"
                out = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model", "gpt-4o-mini"),
                       "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    saved = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "EMBEDDINGS_BACKEND")}
    os.environ.update(OPENAI_API_KEY="sk-local", OPENAI_BASE_URL=f"http://127.0.0.1:{srv.server_port}/v1",
                      OPENAI_API_BASE=f"http://127.0.0.1:{srv.server_port}/v1", EMBEDDINGS_BACKEND="auto")
    try:
        kb, rep = app.build_knowledge_base(FILES)
        assert kb is not None and rep.embedder == app.EMBED_MODEL and not rep.fell_back, rep   # real OpenAIEmbedder used
        assert seen["embeddings"] >= 1
        app.LOG_FILE = Path(tempfile.mkdtemp()) / "answers.txt"
        ev = []
        st1 = app.run_pipeline("How do I reset my password?", kb, lambda a, s, n, t: ev.append((a, s)),
                               search_fn=fake_search)                       # llm_factory=None -> REAL app.make_llm()
        assert st1.error == "" and st1.final_origin == "documents" and ("web", "skipped") in ev, (st1.error, ev)
        st2 = app.run_pipeline("What is the latest stable Python version?", kb, lambda a, s, n, t: None, search_fn=fake_search)
        assert st2.error == "" and st2.final_origin == "web", (st2.error, st2.final_origin, st2.web_raw)
        assert st2.final_sources[0]["url"].startswith("https://www.python.org")
        print(f"      wire test: modes={sorted(seen['modes'])} chat_calls={seen['chat']} tool_calls={seen['tool_calls']} embeddings_calls={seen['embeddings']}")
    finally:
        srv.shutdown()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_line_limit_and_json_parsing():
    assert len(app.limit_lines("\n".join(f"l{i}" for i in range(50)), 10)) == 10
    assert app.parse_json_block('noise {"a": 1} tail') == {"a": 1}
    assert app.parse_json_block('"sufficient": true, "answer": "hi"')["sufficient"] is True
    assert app.parse_json_block("plain text") is None


def test_braces_in_documents_do_not_break_crewai_templating():
    kb, _ = app.build_knowledge_base(FILES)
    state, _ = run("Can I use a custom domain for my workspace?", kb)      # FAQ contains the literal text {custom}
    assert state.error == "", state.error


def test_streamlit_ui_end_to_end():
    from streamlit.testing.v1 import AppTest
    import crewai
    from unittest import mock

    app.LOG_FILE = Path(tempfile.mkdtemp()) / "answers.txt"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    os.environ["SERPER_API_KEY"] = "serper-test"
    cls = make_fake_llm_class()

    class FakeResp:
        status_code = 200
        def json(self):
            return {"organic": [{"title": "Python Releases", "snippet": "3.14 latest", "link": "https://www.python.org/downloads/"}]}

    with mock.patch.object(crewai, "LLM", lambda **kw: cls(model="fake")), \
         mock.patch("requests.post", lambda *a, **k: FakeResp()):
        at = AppTest.from_file(str(HERE / "app.py"), default_timeout=90)
        at.run()
        assert not at.exception, at.exception
        assert [tb.label for tb in at.tabs] == ["Answer", "Knowledge base", "Web search", "Saved record"], "result tabs must show on first load"
        # load sample docs -> KB is built automatically
        next(b for b in at.sidebar.button if b.label == "Load samples").click().run()
        assert not at.exception, at.exception
        assert any("chunks" in m.value for m in at.sidebar.markdown), "KB stats missing"
        assert any("Nimbus_FAQ.txt" in m.value and "Loaded files" in m.value for m in at.sidebar.markdown), "loaded file list missing"
        # question answered from documents
        at.text_area(key="query_text").set_value("How do I reset my password?").run()
        next(b for b in at.button if b.label == "Ask the crew").click().run()
        assert not at.exception, at.exception
        html_all = "\n".join(m.value for m in at.markdown)
        assert "From your documents" in html_all and "Nimbus_Account_Guide.pdf" in html_all
        assert at.session_state["result"].docs, "run must record which documents were searched"
        # changing documents must discard the stale answer
        next(b for b in at.sidebar.button if b.label == "Clear").click().run()
        assert at.session_state["result"] is None
        next(b for b in at.sidebar.button if b.label == "Load samples").click().run()
        assert html_all.count("s-skipped") >= 1 and "s-done" in html_all
        # question that needs the web
        next(b for b in at.button if b.key == "ex_What is the latest stable Python version?").click().run()
        assert not at.exception, at.exception
        html_all = "\n".join(m.value for m in at.markdown)
        assert "From the web" in html_all and "python.org/downloads" in html_all


# ----------------------------------------------------------------------------- live smoke test
def live_smoke():
    os.environ.pop("EMBEDDINGS_BACKEND", None)
    app.load_dotenv(HERE / ".env", override=False)
    print("connections:", app.check_connections())
    kb, rep = app.build_knowledge_base(FILES)
    print("KB:", kb.size, "chunks via", rep.embedder, f"in {rep.seconds:.1f}s")
    for q in ("How do I reset my password?", "What is the latest stable Python version?"):
        st = app.run_pipeline(q, kb, lambda a, s, n, t: print(f"   [{a}] {s} {n}"))
        print(q, "->", st.route, f"{st.total_secs:.1f}s\n", "\n".join(st.final_lines), "\n", st.final_sources)


if __name__ == "__main__":
    if "--live" in sys.argv:
        live_smoke()
        sys.exit(0)
    import time, traceback
    failed = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]:
        t = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print("\nALL TESTS PASSED" if not failed else f"\n{failed} TEST(S) FAILED")
    sys.exit(1 if failed else 0)
