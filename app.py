"""Streamlit chat UI for the Resume Matching Agent."""
import streamlit as st
from langchain_core.messages import HumanMessage
from matching_agent import build_graph

st.set_page_config(page_title="AI Resume Matching Agent", page_icon="🧭", layout="wide")
st.title("🧭 AI Resume Matching Agent")
st.caption("Local synthetic-resume demo · RAG retrieval · evidence-aware ranking")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session"

with st.sidebar:
    st.header("Try a scenario")
    examples = [
        "Find React developers with 3+ years experience",
        "Compare the top 3 matches side by side",
        "Why did Alex Morgan rank higher than Jamie Patel?",
        "Make TypeScript mandatory and rerank everyone",
        "Deep-screen the top 10 and prepare interview questions",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
    st.divider()
    st.caption("Demo data contains synthetic candidate profiles only.")
    if st.button("Clear chat and restart", use_container_width=True):
        st.session_state.chat = []
        st.session_state.graph = build_graph()
        st.session_state.thread_id = "streamlit-session"
        st.rerun()

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Search resumes, refine requirements, compare candidates...")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching and evaluating resume evidence..."):
            try:
                result = st.session_state.graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                answer = result.get("report", "No report generated.")
            except Exception as exc:
                answer = f"**Agent error:** {exc}\n\nCheck the setup steps in README.md."
            st.markdown(answer)
    st.session_state.chat.append({"role": "assistant", "content": answer})
