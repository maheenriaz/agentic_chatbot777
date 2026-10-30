import logging
import streamlit as st
from langgraph_database_backend import (
    workflow, retrive_unique_thread, thread_has_document, thread_document_metadata,
    save_thread_metadata, remove_thread_document,
    clear_thread_checkpoint,
    DEFAULT_USER_ID, get_all_user_memories, delete_user_memory,
    clear_all_user_memories, remember_fact_explicitly,
    routing, store as backend_store,   # regenerate feature ke liye
)
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
import uuid
import re   # file ke top pe agar already nahi hai to add karo
import io

# -------------------
# Confidence Score parsing
# -------------------
# Backend RAG answers ke end mein "[[CONFIDENCE::HIGH/MEDIUM/LOW]]" jaisa hidden
# marker bhejta hai (FAISS similarity score se derive hota hai). Frontend is
# marker ko text se hata kar ek colored badge dikhata hai.
CONFIDENCE_MARKER_PATTERN = re.compile(r'\n*\[\[CONFIDENCE::(HIGH|MEDIUM|LOW)\]\]\s*$')

CONFIDENCE_BADGES = {
    "HIGH": "🟢 **High**",
    "MEDIUM": "🟡 **Medium**",
    "LOW": "🔴 **Low**",
}

def split_confidence(text: str):
    """Text ke end se confidence marker nikalta hai aur (clean_text, label) return karta hai.
    Agar marker na mile to (original_text, None) return hota hai."""
    if not isinstance(text, str):
        return text, None
    match = CONFIDENCE_MARKER_PATTERN.search(text)
    if not match:
        return text, None
    label = match.group(1)
    clean_text = text[:match.start()].rstrip()
    return clean_text, label

def render_message_with_confidence(text: str):
    """Chat bubble ke andar clean text + (agar mojood ho to) confidence badge render karta hai."""
    clean_text, label = split_confidence(text)
    st.markdown(clean_text)
    if label:
        st.caption(CONFIDENCE_BADGES[label])


# -------------------
# Export Chat (TXT / Markdown / PDF)
# -------------------
def _clean_message_content(message: dict) -> str:
    """Assistant messages ke confidence marker ko hata kar plain text deta hai."""
    if message["role"] == "assistant":
        clean_text, _ = split_confidence(message["content"])
        return clean_text
    return message["content"]


def build_chat_text(messages: list, fmt: str = "txt") -> str:
    """Poori chat history ko ek plain text ya markdown string mein format karta hai."""
    lines = []
    for m in messages:
        role = "You" if m["role"] == "user" else "Assistant"
        content = _clean_message_content(m)
        if fmt == "markdown":
            lines.append(f"**{role}:**\n\n{content}\n")
        else:
            lines.append(f"{role}: {content}\n")
    return "\n".join(lines)


def build_chat_pdf(messages: list) -> bytes:
    """Poori chat history se ek simple PDF (bytes) banata hai.
    Requires: pip install fpdf2  (agar installed nahi hai to ImportError raise hoga,
    jo caller UI mein friendly error ke sath dikhata hai)."""
    from fpdf import FPDF  # fpdf2 package - agar missing ho to yahan ImportError aayega

    def _pdf_safe(text: str) -> str:
        # core PDF fonts sirf latin-1 support karte hain - emojis/unicode ko
        # safely replace kar dete hain taake PDF generation crash na ho.
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for m in messages:
        role = "You" if m["role"] == "user" else "Assistant"
        content = _pdf_safe(_clean_message_content(m))

        pdf.set_font("Helvetica", style="B", size=12)
        pdf.multi_cell(0, 8, _pdf_safe(f"{role}:"))
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, content)
        pdf.ln(3)

    raw = pdf.output(dest="S")
    # fpdf2 versions differ: kabhi bytearray/bytes deta hai, kabhi str
    if isinstance(raw, str):
        return raw.encode("latin-1", errors="replace")
    return bytes(raw)


# -------------------
# Regenerate last assistant response
# -------------------
def regenerate_last_response():
    """Graph ke persisted state ka last AI message replace karta hai - routing()
    ko seedha (poore graph ko dobara chalaye bina) call karte hain, isliye
    remember/compress node dobara trigger nahi hote (koi duplicate memory-write
    ya extra compression side-effect nahi hota). skip_cache=True taake semantic
    cache se wohi purana answer wapis na mil jaye."""
    tid = str(st.session_state['thread_id'])
    graph_config = {
        "configurable": {"thread_id": tid, "user_id": DEFAULT_USER_ID}
    }

    current_state = workflow.get_state(graph_config)
    messages = current_state.values.get('messages', [])
    print("messages",messages)
    if not messages or not isinstance(messages[-1], AIMessage):
        st.warning("Regenerate karne ke liye koi assistant response maujood nahi hai.")
        return

    last_ai_message = messages[-1]
    last_ai_id = getattr(last_ai_message, "id", None)

    if last_ai_id is None:
        st.warning("Ye message purane format mein hai (id missing) - regenerate nahi ho sakta.")
        return

    # Naya answer generate karne ke liye state banate hain jisme purana AI
    # message shamil nahi (taake routing() ko last message = user query mile).
    state_for_regen = dict(current_state.values)
    state_for_regen['messages'] = messages[:-1]

    try:
        result = routing(state_for_regen, config=graph_config, store=backend_store, skip_cache=True)
    except Exception as e:
        st.error(f"Regenerate fail hua: {e}")
        return

    new_message = result["messages"][-1]

    # Graph ke persisted checkpoint mein: purana AI message hatao, naya add karo -
    # ye ek hi update mein hota hai (add_messages reducer RemoveMessage + naya
    # message dono ko ek sath process kar leta hai).
    workflow.update_state(
        graph_config,
        {"messages": [RemoveMessage(id=last_ai_id), new_message]},
    )

    # Sidebar/session ki local history bhi update karo
    if st.session_state["message_history"] and st.session_state["message_history"][-1]["role"] == "assistant":
        st.session_state["message_history"][-1] = {"role": "assistant", "content": new_message.content}


def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversations(thread_id):
    state = workflow.get_state({"configurable": {"thread_id": thread_id, "user_id": DEFAULT_USER_ID}})
    return state.values.get('messages', [])

# -------------------
# Session State Init
# -------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_unique_thread()

add_thread(st.session_state['thread_id'])

# -------------------
# Sidebar UI
# -------------------
st.sidebar.title("Welcome Back")

if st.sidebar.button("New Chat"):
    reset_chat()
    save_thread_metadata(str(st.session_state['thread_id']), {"thread_name": "New Chat"})

st.sidebar.markdown("---")
st.sidebar.header("My Conversations")

# -------------------
# Conversation Search
# -------------------
# Default: sirf thread ke naam (title) pe filter karta hai - fast hai, koi
# extra DB/checkpoint load nahi hoti.
# "Also search inside messages" on karne par har thread ki poori chat load
# kar ke content bhi check karta hai - thoda slow ho sakta hai agar bohot
# saari threads hon, isliye opt-in checkbox hai.
search_query = st.sidebar.text_input(
    "🔍 Search conversations", key="thread_search", placeholder="Type to filter..."
)
search_in_messages = st.sidebar.checkbox(
    "Also search inside messages", value=False, key="thread_search_deep",
    help="Slower - checks full chat content too, not just the conversation title."
)

# Pehle har thread ka display label precompute kar lete hain (name-filter ke liye)
_thread_entries = []
for tid in st.session_state['chat_threads']:
    meta = thread_document_metadata(str(tid))
    # DB se thread_name milta hai - agar nahi mila to fallback short id
    label = meta.get("thread_name") or (str(tid)[:8] + "...")
    label = label[:30]
    if thread_has_document(str(tid)):
        label = "📄 " + label
    _thread_entries.append((tid, label))

if search_query.strip():
    q = search_query.strip().lower()
    filtered_entries = []
    for tid, label in _thread_entries:
        if q in label.lower():
            filtered_entries.append((tid, label))
            continue
        if search_in_messages:
            try:
                msgs = load_conversations(tid)
                content_match = any(
                    isinstance(m.content, str) and q in m.content.lower() for m in msgs
                )
            except Exception:
                content_match = False
            if content_match:
                filtered_entries.append((tid, label))

    if not filtered_entries:
        st.sidebar.caption("Not Found")
else:
    filtered_entries = _thread_entries

for tid, label in filtered_entries:
    if st.sidebar.button(label, key=str(tid)):
        st.session_state['thread_id'] = tid
        messages = load_conversations(tid)
        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages

# -------------------
# Sidebar — Long-Term Memory viewer
# -------------------
st.sidebar.markdown("---")
with st.sidebar.expander("🧠 What I remember about you"):
    memories = get_all_user_memories(DEFAULT_USER_ID)

    if not memories:
        st.caption("No memory found")
    else:
        for mem in memories:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"- {mem['text']}")
            with col2:
                if st.button("🗑️", key=f"del_{mem['key']}"):
                    delete_user_memory(mem['key'], DEFAULT_USER_ID)
                    st.rerun()

    st.markdown("---")
    # manual_memory = st.text_input("Manually remember something:", key="manual_memory_input")
    # if st.button("Save memory", key="save_manual_memory"):
    #     if manual_memory.strip():
    #         remember_fact_explicitly(manual_memory.strip(), DEFAULT_USER_ID)
    #         st.rerun()

    # if memories and st.button("Clear all memories", key="clear_all_memories"):
    #     clear_all_user_memories(DEFAULT_USER_ID)
    #     st.rerun()

# -------------------
# Sidebar — Export Chat
# -------------------
st.sidebar.markdown("---")
st.sidebar.header("📤 Export Chat")

if not st.session_state["message_history"]:
    st.sidebar.caption("No chats found to export")
else:
    export_format = st.sidebar.selectbox(
        "Format", ["TXT", "Markdown", "PDF"], key="export_format"
    )

    if export_format == "TXT":
        st.sidebar.download_button(
            "⬇️ Download chat",
            data=build_chat_text(st.session_state["message_history"], fmt="txt"),
            file_name="chat_export.txt",
            mime="text/plain",
        )
    elif export_format == "Markdown":
        st.sidebar.download_button(
            "⬇️ Download chat",
            data=build_chat_text(st.session_state["message_history"], fmt="markdown"),
            file_name="chat_export.md",
            mime="text/markdown",
        )
    else:  # PDF
        try:
            pdf_bytes = build_chat_pdf(st.session_state["message_history"])
            st.sidebar.download_button(
                "⬇️ Download chat",
                data=pdf_bytes,
                file_name="chat_export.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.sidebar.error("PDF export ke liye pehle install karo: `pip install fpdf2`")
        except Exception as e:
            st.sidebar.error(f"PDF export fail hua: {e}")

# -------------------
# Main Chat UI
# -------------------
st.title("🤖 Agentic AI Chatbot")
st.caption("An intelligent AI assistant powered by agents, memory, and RAG.")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_message_with_confidence(message["content"])
        else:
            st.markdown(message["content"])

# -------------------
# Regenerate last response
# -------------------
# Sirf tab dikhta hai jab last message assistant ki ho (yaani koi response
# maujood hai jo dobara generate ki ja sake).
if st.session_state["message_history"] and st.session_state["message_history"][-1]["role"] == "assistant":
    if st.button("↻ Regenerate", key="regenerate_btn"):
        with st.spinner("🤖 Regenerating..."):
            regenerate_last_response()
        st.rerun()

# upload_container = st.container()
# with upload_container:
#     current_tid = str(st.session_state['thread_id'])
#     uploaded_file = st.file_uploader("Attach a PDF", type=["pdf"])

#     if uploaded_file is not None:
#         # st.success(f"✅ {uploaded_file.name} uploaded")
#         result = ingest_pdf(uploaded_file.read(), current_tid, uploaded_file.name)
#         # if result["success"]:
#         #     st.info(f"📄 {uploaded_file.name} ready! ({result['pages']} pages, {result['chunks']} chunks)")
#         # else:
#         #     st.error(f"❌ Error: {result['error']}")
#     else:
#         # st.warning("📄 No PDF attached (removed or not uploaded)")
#         if thread_has_document(current_tid):
#             remove_thread_document(current_tid)
#             clear_thread_checkpoint(current_tid)
#             # st.info("🗑️ Document context cleared for this thread")

# Sirf yehi nodes hain jinke LLM tokens user ko chat mein dikhne chahiye -
# koi bhi aur node (remember, tools, compress) ya internal helper call
# (classifier, summarizer, memory extractor) hamesha yahan block hone chahiye.
# Yeh "deny by node-name + tag" dono checks combine karta hai taake future mein
# koi naya internal node/helper add ho to accidental leak na ho.
#
# NOTE: "guardrail_classifier" is included here because guardrails.py's
# check_guardrails() calls llm_classifier.invoke(..., config={"tags": ["guardrail_classifier"]})
# from *inside* the routing node. Since that call happens inside the (visible)
# "routing" node, only the tag-filter below stops its raw SAFE/CONFIDENTIAL/
# PROMPT_LEAK/HARMFUL/INJECTION classification token from leaking into the chat.
HIDDEN_NODES = {"tools", "remember", "compress"}
HIDDEN_TAGS = {"intent_classifier", "summarizer", "memory_extractor", "guardrail_classifier"}

user_input = st.chat_input("Type here...")

if user_input:
    current_tid = str(st.session_state['thread_id'])

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # if uploaded_file is not None:
    #     if not thread_has_document(current_tid):
    #         result = ingest_pdf(
    #             file_bytes=uploaded_file.read(),
    #             thread_id=current_tid,
    #             filename=uploaded_file.name
    #         )
    # else:
    #     remove_thread_document(current_tid)kk

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        # -------------------
        # Typing indicator
        # -------------------
        # Streaming shuru hone se pehle (jab tak backend routing + retrieval +
        # model ka pehla token generate nahi ho jata) ek "thinking" indicator
        # dikhate hain, taake user ko lage app hang nahi hui balki kaam ho raha hai.
        # Jaise hi pehla real content token aata hai, ye automatically replace ho
        # jata hai (neeche ka placeholder.markdown(streamed_text) hi isko overwrite
        # kar deta hai).
        placeholder.markdown("🤖 *Thinking...*")

        stream = workflow.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={
                # ✅ user_id zaroori hai — remember_node aur routing node isko use karte hain
                # long-term memory ko sahi user ke namespace mein read/write karne ke liye
                "configurable": {
                    "thread_id": st.session_state['thread_id'],
                    "user_id": DEFAULT_USER_ID
                },
                "metadata": {"thread_id": st.session_state['thread_id']},
                "run_name": "chat_turn"
            },
            stream_mode="messages"
        )
        # print(f"[STREAMING RESPONSE] {stream}")
        sources_cutoff = False

        for chunk, metadata in stream:
            content = getattr(chunk, "content", "")
            if not content:
                continue
            if content.strip().lower() in ["tool", "document", "yes", "no"]:
                continue

            node_name = metadata.get("langgraph_node")
            if node_name in HIDDEN_NODES:
                continue

            chunk_tags = set(metadata.get("tags", []) or [])
            if chunk_tags & HIDDEN_TAGS:
                continue

            if sources_cutoff:
                continue   # sources shuru hone ke baad aane wale tokens ignore karo

            streamed_text += content

            # ✅ "Sources:" / "References:" pattern check karo
            match = re.search(r'\s*\**\s*(Sources?|References?|Citations?)\s*\**\s*:', streamed_text, re.IGNORECASE)
            if match:
                streamed_text = streamed_text[:match.start()].rstrip()
                sources_cutoff = True

            placeholder.markdown(streamed_text)

        # Stream khatam - ab graph ke persisted state se ASLI final message
        # nikalte hain (streamed_text sirf live token-preview ke liye tha).
        # Do wajah se ye zaroori hai:
        #  1) RAG answers ke end mein ek hidden confidence marker hota hai jo
        #     model ke tokens ke sath stream NAHI hota (kyunki wo llm.invoke()
        #     ke baad, post-processing mein, answer ke sath jode jate hain) -
        #     isliye asli marker sirf persisted state ke message mein milega.
        #  2) Semantic-cache-hit wale case mein koi model call hi nahi hoti,
        #     isliye us case mein streamed_text khaali reh jata - persisted
        #     state se hi actual cached answer milta hai.
        #  3) Guardrail short-circuit (confidential/off-topic/harmful/etc) mein
        #     bhi koi llm.invoke() call nahi hoti - AIMessage seedha routing()
        #     se return hoti hai - is case mein bhi streamed_text khaali rehta
        #     hai aur persisted state hi ka canned response dikhta hai.
        graph_config = {
            "configurable": {
                "thread_id": st.session_state['thread_id'],
                "user_id": DEFAULT_USER_ID
            }
        }
        try:
            final_state = workflow.get_state(graph_config)
            final_messages = final_state.values.get('messages', [])
            final_answer = final_messages[-1].content if final_messages else streamed_text
        except Exception as e:
            logging.error(f"[FINAL STATE] failed to fetch, falling back to streamed text: {e}")
            final_answer = streamed_text

        if not isinstance(final_answer, str) or not final_answer.strip():
            final_answer = streamed_text

        placeholder.empty()
        with placeholder.container():
            render_message_with_confidence(final_answer)

        st.session_state["message_history"].append({
                    "role": "assistant",
                    "content": final_answer
        })

    # Pehla user message ke baad thread ka naam DB mein save karo (persistent), phir rerun karo
    # taake sidebar turant updated naam aur naya memory (agar koi save hua ho) dikhaye
    if len(st.session_state["message_history"]) == 2:
        try:
            save_thread_metadata(current_tid, {"thread_name": user_input[:30]})
        except Exception as e:
            logging.error(f"[THREAD NAME] failed to save: {e}")
        st.rerun()