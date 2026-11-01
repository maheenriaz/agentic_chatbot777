import base64
import logging
import streamlit as st
from langgraph_database_backend import (
    workflow, retrive_unique_thread, thread_has_document, thread_document_metadata,
    save_thread_metadata, clear_thread_checkpoint,
    DEFAULT_USER_ID, get_all_user_memories, delete_user_memory,
    routing, store as backend_store,
    get_form_path, split_download_marker, FORMS,
)
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
import uuid
import re

st.set_page_config(page_title="PeopleDesk AI", page_icon="🏦", layout="wide")

# =====================================================================
# DESIGN TOKENS + GLOBAL CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --navy:#0B1F4B; --royal:#1D4ED8; --royal-soft:#EEF3FF; --ink:#1B2437; --muted:#667085;
  --line:#E3E8F0; --surface:#FFFFFF; --canvas:#F5F7FA; --radius:14px;
  --shadow:0 1px 2px rgba(11,31,75,.06),0 4px 14px rgba(11,31,75,.05);
}
html,body,[class*="css"],.stApp{font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink);}
.stApp{background:var(--canvas);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{max-width:860px;padding-top:1.5rem;padding-bottom:7rem;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .stButton>button{
  width:100%;justify-content:flex-start;text-align:left;border:none;background:transparent;
  color:var(--ink);border-radius:10px;padding:.5rem .75rem;font-weight:400;transition:background .15s;}
section[data-testid="stSidebar"] .stButton>button:hover{background:var(--canvas);}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:var(--royal-soft);color:var(--royal);font-weight:500;}
.brand{display:flex;align-items:center;gap:.7rem;margin-bottom:.25rem;}
.brand-mark{width:36px;height:36px;border-radius:10px;background:var(--navy);color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:600;font-size:15px;}
.brand-name{font-weight:600;font-size:1.05rem;color:var(--navy);line-height:1.1;}
.brand-sub{font-size:.78rem;color:var(--muted);}
.profile{display:flex;align-items:center;gap:.7rem;padding:.7rem;border:1px solid var(--line);border-radius:12px;margin-top:1rem;}
.avatar{width:34px;height:34px;border-radius:50%;background:var(--royal-soft);color:var(--royal);
  display:flex;align-items:center;justify-content:center;font-weight:600;font-size:13px;}
.profile b{font-size:.88rem;display:block;} .profile span{font-size:.75rem;color:var(--muted);}

/* ---------- header ---------- */
.status{display:inline-flex;align-items:center;gap:.45rem;font-size:.8rem;color:var(--muted);
  background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:.2rem .7rem;}
.status i{width:7px;height:7px;border-radius:50%;background:#16A34A;display:inline-block;}
.hero h1{font-size:2.1rem;font-weight:600;color:var(--navy);letter-spacing:-.02em;margin:.9rem 0 .4rem;}
.hero p{color:var(--muted);font-size:1rem;max-width:560px;margin:0;}
.greet{margin:1.8rem 0 .2rem;font-size:1.15rem;font-weight:500;color:var(--navy);}
.greet-sub{color:var(--muted);font-size:.92rem;margin-bottom:1rem;}

/* ---------- suggestion cards (Streamlit buttons in main area) ---------- */
.main .stButton>button{
  border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:var(--radius);
  box-shadow:var(--shadow);text-align:left;padding:.9rem 1rem;height:100%;min-height:92px;
  white-space:normal;justify-content:flex-start;transition:border-color .15s,transform .15s;}
.main .stButton>button:hover{border-color:var(--royal);color:var(--ink);transform:translateY(-1px);}
.main .stButton>button p{font-size:.88rem;line-height:1.4;text-align:left;}
.main .stButton>button p:first-line{font-weight:600;color:var(--navy);}
div.quick .stButton>button{min-height:0;padding:.35rem .8rem;border-radius:999px;box-shadow:none;height:auto;}

/* ---------- chat bubbles ---------- */
div[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--line);
  border-radius:var(--radius);padding:1rem 1.1rem;margin-bottom:.8rem;box-shadow:var(--shadow);}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  flex-direction:row-reverse;background:var(--royal-soft);border-color:#D6E2FF;margin-left:15%;text-align:right;}
div[data-testid="stChatMessage"] table{font-size:.88rem;}
.conf{font-size:.78rem;color:var(--muted);margin-top:.3rem;}

            /* Regenerate button */
div[data-testid="stChatMessage"] button[kind="secondary"] {
    border: 1px solid #D0D5DD;
    background: #FFFFFF;
    color: #344054;
    border-radius: 20px;
    padding: 5px 14px;
    min-height: 34px;
    font-size: 13px;
    font-weight: 500;
    box-shadow: none;
    width: auto;
}

div[data-testid="stChatMessage"] button[kind="secondary"]:hover {
    border-color: #1D4ED8;
    color: #1D4ED8;
    background: #EEF3FF;
}
            
/* ---------- document card ---------- */
.doc{display:flex;gap:.9rem;align-items:center;border:1px solid var(--line);border-radius:12px;
  padding:.9rem 1rem;margin:.6rem 0 .4rem;background:var(--canvas);}
.doc-ic{width:44px;height:52px;border-radius:8px;background:#fff;border:1px solid var(--line);
  display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:600;color:#B42318;}
.doc b{display:block;color:var(--navy);} .doc small{color:var(--muted);display:block;}

/* ---------- typing indicator ---------- */
.typing{display:inline-flex;gap:5px;padding:.4rem 0;}
.typing span{width:7px;height:7px;border-radius:50%;background:var(--royal);opacity:.35;animation:blink 1.2s infinite;}
.typing span:nth-child(2){animation-delay:.2s;} .typing span:nth-child(3){animation-delay:.4s;}
@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
@media (prefers-reduced-motion:reduce){.typing span{animation:none;opacity:.7}}

/* ---------- composer ---------- */
div[data-testid="stChatInput"]{border:1px solid var(--line);border-radius:18px;background:var(--surface);box-shadow:var(--shadow);}
div[data-testid="stChatInput"]:focus-within{border-color:var(--royal);}
@media (max-width:640px){
  .hero h1{font-size:1.6rem;}
  div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){margin-left:0;}
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONFIDENCE / DOWNLOAD MARKERS (unchanged backend contract)
# =====================================================================
CONFIDENCE_MARKER_PATTERN = re.compile(r'\n*\[\[CONFIDENCE::(HIGH|MEDIUM|LOW)\]\]\s*$')
CONFIDENCE_BADGES = {"HIGH": "🟢 High confidence", "MEDIUM": "🟡 Medium confidence", "LOW": "🔴 Low confidence"}

FORM_DESCRIPTIONS = {
    "onboarding": "Official form for new employee joining and documentation.",
    "offboarding": "Official form for employee exit and clearance procedures.",
}


def split_confidence(text: str):
    if not isinstance(text, str):
        return text, None
    m = CONFIDENCE_MARKER_PATTERN.search(text)
    if not m:
        return text, None
    return text[:m.start()].rstrip(), m.group(1)

def render_document_card(form_type: str, key_suffix: str):
    """Document card backed by the real PDF via get_form_path()."""
    path = get_form_path(form_type)
    if not path:
        return
    info = FORMS[form_type]
    st.markdown(
        f"""<div class="doc"><div class="doc-ic">PDF</div>
        <div><b>{info['label']}</b><small>PDF Document</small>
        <small>{FORM_DESCRIPTIONS.get(form_type, '')}</small></div></div>""",
        unsafe_allow_html=True,
    )
    with open(path, "rb") as f:
        pdf_bytes = f.read()

    col_dl, _ = st.columns([3, 7])
    with col_dl:
        st.download_button("⬇ Download PDF", data=pdf_bytes, file_name=info["file"],
                           mime="application/pdf", key=f"dl_{form_type}_{key_suffix}",
                           use_container_width=True)

    with st.expander("👁 View PDF"):
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="600" style="border:1px solid #E3E8F0;border-radius:10px;"></iframe>',
            unsafe_allow_html=True,
        )

def clean_text(message: dict) -> str:
    if message["role"] == "assistant":
        t, _ = split_download_marker(message["content"])
        t, _ = split_confidence(t)
        return t
    return message["content"]

def render_assistant(text: str, key_suffix: str, is_last: bool = False):
    """Message body + document card + source line + actions (copy / feedback / regenerate)."""
    text, form_type = split_download_marker(text)
    body, label = split_confidence(text)
    st.markdown(body)
    if form_type:
        render_document_card(form_type, key_suffix)
    if label:
        st.markdown(f'<div class="conf">Source: HR Policy Document · {CONFIDENCE_BADGES[label]}</div>',
                    unsafe_allow_html=True)

    if is_last:
        col1, _ = st.columns([2, 8])
        with col1:
            if st.button("↻ Regenerate", key=f"regen_{key_suffix}", type="secondary"):
                with st.spinner("Regenerating..."):
                    regenerate_last_response()
                st.rerun()

# =====================================================================
# EXPORT (TXT / Markdown / PDF)
# =====================================================================
def build_chat_text(messages: list, fmt: str = "txt") -> str:
    lines = []
    for m in messages:
        role = "You" if m["role"] == "user" else "Assistant"
        content = clean_text(m)
        lines.append(f"**{role}:**\n\n{content}\n" if fmt == "markdown" else f"{role}: {content}\n")
    return "\n".join(lines)


def build_chat_pdf(messages: list) -> bytes:
    from fpdf import FPDF  # pip install fpdf2

    def safe(t: str) -> str:
        return t.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for m in messages:
        role = "You" if m["role"] == "user" else "Assistant"
        pdf.set_font("Helvetica", style="B", size=12)
        pdf.multi_cell(0, 8, safe(f"{role}:"))
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, safe(clean_text(m)))
        pdf.ln(3)
    raw = pdf.output(dest="S")
    return raw.encode("latin-1", errors="replace") if isinstance(raw, str) else bytes(raw)


# =====================================================================
# REGENERATE (same logic as before)
# =====================================================================
def regenerate_last_response():
    tid = str(st.session_state['thread_id'])
    graph_config = {"configurable": {"thread_id": tid, "user_id": DEFAULT_USER_ID}}
    current_state = workflow.get_state(graph_config)
    messages = current_state.values.get('messages', [])
    if not messages or not isinstance(messages[-1], AIMessage):
        st.warning("There is no assistant response to regenerate.")
        return
    last_ai_id = getattr(messages[-1], "id", None)
    if last_ai_id is None:
        st.warning("This message was saved in an older format and can't be regenerated.")
        return
    state_for_regen = dict(current_state.values)
    state_for_regen['messages'] = messages[:-1]
    try:
        result = routing(state_for_regen, config=graph_config, store=backend_store, skip_cache=True)
    except Exception as e:
        st.error(f"Couldn't regenerate the response: {e}")
        return
    new_message = result["messages"][-1]
    workflow.update_state(graph_config, {"messages": [RemoveMessage(id=last_ai_id), new_message]})
    hist = st.session_state["message_history"]
    if hist and hist[-1]["role"] == "assistant":
        hist[-1] = {"role": "assistant", "content": new_message.content}


# =====================================================================
# THREAD HELPERS + SESSION STATE
# =====================================================================
def generate_thread_id():
    return uuid.uuid4()


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    tid = generate_thread_id()
    st.session_state['thread_id'] = tid
    add_thread(tid)
    st.session_state['message_history'] = []
    save_thread_metadata(str(tid), {"thread_name": "New Chat"})


def load_conversations(thread_id):
    state = workflow.get_state({"configurable": {"thread_id": thread_id, "user_id": DEFAULT_USER_ID}})
    return state.values.get('messages', [])


def ask(prompt: str):
    st.session_state["pending_query"] = prompt


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_unique_thread()
add_thread(st.session_state['thread_id'])

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-mark">PD</div>
        <div><div class="brand-name">PeopleDesk AI</div>
        <div class="brand-sub">Intelligent HR Assistant</div></div></div>""", unsafe_allow_html=True)
    st.write("")
    if st.button("＋  New chat", key="new_chat", type="primary"):
        reset_chat()
        st.rerun()

    search_query = st.text_input("Search conversations", key="thread_search",
                                 placeholder="Search conversations", label_visibility="collapsed")
    search_in_messages = st.checkbox("Also search inside messages", value=False, key="thread_search_deep",
                                     help="Slower: checks full chat content, not just titles.")

    entries = []
    for tid in st.session_state['chat_threads']:
        meta = thread_document_metadata(str(tid))
        label = (meta.get("thread_name") or (str(tid)[:8] + "..."))[:30]
        if thread_has_document(str(tid)):
            label = "📄 " + label
        entries.append((tid, label))

    if search_query.strip():
        q = search_query.strip().lower()
        shown = []
        for tid, label in entries:
            if q in label.lower():
                shown.append((tid, label))
            elif search_in_messages:
                try:
                    if any(isinstance(m.content, str) and q in m.content.lower() for m in load_conversations(tid)):
                        shown.append((tid, label))
                except Exception:
                    pass
        if not shown:
            st.caption("No conversations match your search.")
    else:
        shown = entries

    st.caption("Recent conversations")
    for tid, label in shown:
        active = str(tid) == str(st.session_state['thread_id'])
        if st.button(label, key=f"thr_{tid}", type="primary" if active else "secondary"):
            st.session_state['thread_id'] = tid
            st.session_state['message_history'] = [
                {'role': 'user' if isinstance(m, HumanMessage) else 'assistant', 'content': m.content}
                for m in load_conversations(tid)
            ]
            st.rerun()

    if st.session_state["message_history"]:
        if st.button("🗑  Clear this conversation", key="clear_conv"):
            clear_thread_checkpoint(str(st.session_state['thread_id']))
            st.session_state["message_history"] = []
            st.rerun()

    with st.expander("⚙️  Settings"):
        st.caption("What I remember about you")
        memories = get_all_user_memories(DEFAULT_USER_ID)
        if not memories:
            st.caption("No saved memories yet.")
        for mem in memories:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"- {mem['text']}")
            if c2.button("✕", key=f"del_{mem['key']}"):
                delete_user_memory(mem['key'], DEFAULT_USER_ID)
                st.rerun()

        st.markdown("---")
        st.caption("Export this chat")
        if not st.session_state["message_history"]:
            st.caption("Nothing to export yet.")
        else:
            fmt = st.selectbox("Format", ["TXT", "Markdown", "PDF"], key="export_format")
            hist = st.session_state["message_history"]
            if fmt == "TXT":
                st.download_button("Download chat", build_chat_text(hist), "chat_export.txt", "text/plain")
            elif fmt == "Markdown":
                st.download_button("Download chat", build_chat_text(hist, "markdown"), "chat_export.md", "text/markdown")
            else:
                try:
                    st.download_button("Download chat", build_chat_pdf(hist), "chat_export.pdf", "application/pdf")
                except ImportError:
                    st.error("PDF export needs fpdf2. Run: pip install fpdf2")
                except Exception as e:
                    st.error(f"PDF export failed: {e}")

    st.markdown("""<div class="profile"><div class="avatar">EM</div>
        <div><b>Employee</b><span>Signed in</span></div></div>""", unsafe_allow_html=True)

# =====================================================================
# MAIN AREA
# =====================================================================
# chat_input renders fixed at the bottom; reading it first lets the welcome
# screen disappear on the same run in which a message is submitted.
typed = st.chat_input("Ask anything about HR policies...")
user_input = typed or st.session_state.pop("pending_query", None)

st.markdown('<span class="status"><i></i>HR Knowledge Assistant</span>', unsafe_allow_html=True)

SUGGESTIONS = [
    ("📅", "Leave Management", "How many annual leaves am I entitled to?"),
    ("💳", "Employee Financing", "What are the eligibility criteria for a personal loan?"),
    ("🧾", "Onboarding", "I am a new employee. What documents do I need?"),
    ("🚪", "Offboarding", "Can you provide the employee offboarding form?"),
    ("🕘", "Attendance", "What are the official office timings?"),
    ("📘", "HR Policies", "What is the maternity leave policy?"),
]
QUICK = [
    ("Leave policies", "What are the leave policies?"),
    ("Employee benefits", "What employee benefits are available?"),
    ("Employee financing", "What employee financing options are available?"),
    ("Working hours", "What are the attendance and working hours rules?"),
    ("Onboarding documents", "What are the steps to onboard a new employee?"),
    ("Offboarding documents", "Can you provide the employee offboarding form?"),
    ("Code of conduct", "What does the code of conduct say?"),
]

if not st.session_state["message_history"] and not user_input:
    st.markdown("""<div class="hero"><h1>Your HR, Simplified.</h1>
        <p>Your intelligent assistant for HR policies, employee benefits, documents and workplace support.</p></div>
        <div class="greet">Hello! How can I help you today?</div>
        <div class="greet-sub">Ask me anything about HR policies, leave management, employee benefits,
        onboarding, offboarding and more.</div>""", unsafe_allow_html=True)

    for row in range(0, len(SUGGESTIONS), 3):
        cols = st.columns(3)
        for col, (icon, title, prompt) in zip(cols, SUGGESTIONS[row:row + 3]):
            with col:
                st.button(f"{icon}  {title}\n\n{prompt}", key=f"sg_{title}",
                          on_click=ask, args=(prompt,), use_container_width=True)

    st.caption("Quick access")
    st.markdown('<div class="quick">', unsafe_allow_html=True)
    qcols = st.columns(4)
    for i, (label, prompt) in enumerate(QUICK):
        with qcols[i % 4]:
            st.button(label, key=f"qk_{label}", on_click=ask, args=(prompt,), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- history ----------
history = st.session_state["message_history"]
for i, message in enumerate(history):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message["content"], key_suffix=f"hist_{i}", is_last=(i == len(history) - 1))
        else:
            st.markdown(message["content"])

# ---------- new turn ----------
if user_input:
    current_tid = str(st.session_state['thread_id'])
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown('<div class="typing"><span></span><span></span><span></span></div>',
                             unsafe_allow_html=True)
        error = None
        try:
            workflow.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={
                    "configurable": {"thread_id": st.session_state['thread_id'], "user_id": DEFAULT_USER_ID},
                    "metadata": {"thread_id": st.session_state['thread_id']},
                    "run_name": "chat_turn",
                },
            )
        except Exception as e:
            logging.error(f"[INVOKE] failed: {e}")
            error = str(e)

        final_state = workflow.get_state(
            {"configurable": {"thread_id": st.session_state['thread_id'], "user_id": DEFAULT_USER_ID}}
        )
        msgs = final_state.values.get('messages', [])
        final_answer = msgs[-1].content if msgs else ""
        if error and not isinstance(final_answer, str):
            final_answer = ""
        if error and not final_answer:
            final_answer = "Something went wrong while generating the answer. Please try again."

        placeholder.empty()
        with placeholder.container():
            render_assistant(final_answer, key_suffix=f"new_{len(history)}", is_last=True)

        history.append({"role": "assistant", "content": final_answer})

    if len(history) == 2:
        try:
            save_thread_metadata(current_tid, {"thread_name": user_input[:30]})
        except Exception as e:
            logging.error(f"[THREAD NAME] failed to save: {e}")
    st.rerun()