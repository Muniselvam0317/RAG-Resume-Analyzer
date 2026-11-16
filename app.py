import streamlit as st

import config
from rag.pipeline import ResumeRAG

st.set_page_config(page_title="Resume Analyzer (RAG)", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and index...")
def get_rag() -> ResumeRAG:
    return ResumeRAG()


rag = get_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📁 Resumes")
    files = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.button("⚙️ Process resumes", type="primary", disabled=not files,
                 use_container_width=True):
        bar = st.progress(0.0, text="Starting...")
        res = rag.ingest(
            [(f.name, f.getvalue()) for f in files],
            progress=lambda p, name: bar.progress(p, text=f"Indexed {name}"),
        )
        bar.empty()
        st.success(f"Added {len(res['added'])} resume(s)")
        if res["skipped"]:
            st.info("Skipped: " + "; ".join(res["skipped"]))
        if res["failed"]:
            st.error("Failed:\n" + "\n".join(res["failed"]))
        st.session_state.uploader_key += 1  # reset uploader after processing

    st.divider()
    cands = rag.store.candidates()
    st.metric("Resumes indexed", len(cands))
    st.metric("Chunks in FAISS", rag.store.size)
    if cands:
        with st.expander("Indexed candidates"):
            for name, fname in cands:
                st.write(f"• **{name}** — {fname}")
    if st.button("🗑️ Clear index & chat", use_container_width=True):
        rag.store.clear()
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------- main
st.title("📄 Resume Analyzer")
st.caption("Ask anything about the uploaded resumes. Answers are grounded in the resume text.")

if not rag.store.size:
    st.info("Upload resumes in the sidebar and click **Process resumes** to begin.")



def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)} excerpts)"):
        for s in sources:
            st.markdown(f"**{s['candidate']}** · `{s['file']}` · similarity {s['score']}")
            st.text(s["text"][:500])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

if question := st.chat_input("Ask about the resumes..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            stream, sources = rag.ask(question, history)
            answer = st.write_stream(stream)
            show_sources(sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error(f"Error: {e}")
