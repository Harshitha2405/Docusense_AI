import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from ocr.extractor import extract_text
from router import detect_document_type, get_document_label
from pipelines.id_card import analyse_id_document
from pipelines.financial import build_rag_chain, extract_financial_summary, extract_bank_summary
from pipelines.contract import analyse_contract, analyse_invoice
from utils.verifier import cross_verify_documents
from utils.report_gen import generate_pdf_report

st.set_page_config(page_title="DocuSense AI", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html,body,.stApp{background:#f1f5f9;font-family:'Inter',sans-serif;color:#0f172a;}
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e2e8f0;}
p,div,span,label,li,td,th{font-size:14px!important;color:#0f172a!important;}

/* ── Topbar ── */
.topbar{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
        padding:1rem 1.6rem;margin-bottom:1.2rem;
        box-shadow:0 1px 3px rgba(0,0,0,.06);
        display:flex;align-items:center;gap:14px;}
.t-title{font-size:1.35rem!important;font-weight:700!important;color:#0f172a!important;margin:0;}
.t-sub{font-size:13px!important;color:#64748b!important;margin:2px 0 0;}

/* ── Card ── */
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
      padding:1.2rem 1.4rem;margin-bottom:.9rem;
      box-shadow:0 1px 3px rgba(0,0,0,.05);}

/* ── Labels ── */
.sec{font-size:10px!important;font-weight:700!important;text-transform:uppercase;
     letter-spacing:.09em;color:#94a3b8!important;margin:0 0 8px;}
.badge{display:inline-flex;align-items:center;gap:5px;
       background:#eff6ff;color:#1d4ed8!important;
       border:1px solid #bfdbfe;padding:3px 12px;border-radius:20px;
       font-size:12px!important;font-weight:600!important;margin-bottom:10px;}

/* ── Fields table ── */
.ftable{width:100%;border-collapse:collapse;}
.ftable tr{border-bottom:1px solid #f1f5f9;}
.ftable tr:last-child{border-bottom:none;}
.ftable td{padding:10px 6px;vertical-align:top;}
.ftable td.fl{font-size:12px!important;color:#475569!important;
              font-weight:500!important;width:44%;white-space:nowrap;}
.ftable td.fv{font-size:14px!important;color:#0f172a!important;
              font-weight:700!important;text-align:right;word-break:break-word;}

/* ── Raw text panel ── */
.raw-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;}
.raw-title{font-size:10px!important;font-weight:700!important;
           text-transform:uppercase;letter-spacing:.08em;color:#94a3b8!important;}
.raw-ok{font-size:11px!important;color:#16a34a!important;font-weight:600!important;}
.raw-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
         padding:14px 16px;font-family:'Courier New',monospace;
         font-size:13px!important;color:#1e293b!important;
         line-height:1.85;white-space:pre-wrap;word-break:break-word;
         height:460px;overflow-y:auto;}

/* ── Status ── */
.v-ok {font-size:13px!important;color:#16a34a!important;display:flex;align-items:center;gap:5px;margin:4px 0;}
.v-err{font-size:13px!important;color:#dc2626!important;display:flex;align-items:center;gap:5px;margin:4px 0;}
.v-warn{font-size:13px!important;color:#d97706!important;display:flex;align-items:center;gap:5px;margin:4px 0;}

/* ── Sidebar ── */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 14px;}
.stat-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px;text-align:center;}
.stat-n{font-size:20px!important;font-weight:700!important;color:#2563eb!important;}
.stat-l{font-size:11px!important;color:#94a3b8!important;margin-top:2px;}
.nav-item{font-size:13px!important;color:#334155!important;padding:3px 0;
          display:flex;align-items:center;gap:8px;}

/* ── Chat ── */
.msg-user{background:#2563eb;color:#fff!important;padding:10px 16px;
          border-radius:18px 18px 4px 18px;margin:8px 0 8px auto;
          max-width:70%;font-size:14px!important;width:fit-content;
          line-height:1.5;font-weight:500!important;}
.msg-bot{background:#fff;color:#0f172a!important;border:1px solid #e2e8f0;
         padding:10px 16px;border-radius:4px 18px 18px 18px;
         margin:8px 0;max-width:80%;font-size:14px!important;
         line-height:1.75;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.chat-hint{text-align:center;color:#94a3b8!important;
           font-size:13px!important;padding:1.5rem;line-height:2;}

/* ── Misc ── */
.divider{border:none;border-top:1px solid #e2e8f0;margin:1rem 0;}
[data-testid="stFileUploader"]{background:#fff;border:2px dashed #bfdbfe;border-radius:12px;}
#MainMenu,footer,.stDeployButton{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
defaults = {
    "processed_docs":   [],
    "rag_chain":        None,
    "chat_history":     [],
    "current_raw_text": "",
    "all_raw_texts":    [],
    "raw_store":        {},
    "fields_store":     {},
    "doctype_store":    {},
    "doclabel_store":   {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ID_TYPES = ["aadhaar","pan","voter_id","passport","driving_licence","salary_slip"]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 DocuSense AI")
    st.caption("Document Intelligence Platform")
    st.divider()
    st.markdown("<div class='sec'>Supported Documents</div>", unsafe_allow_html=True)
    for icon, name in [
        ("🪪","Aadhaar Card"),("💳","PAN Card"),("🗳️","Voter ID"),
        ("📗","Passport"),("🚗","Driving Licence"),("🏦","Bank Statement"),
        ("📊","P&L / Balance Sheet"),("📝","Contract / NDA"),
        ("🧾","Invoice / Bill"),("💰","Salary Slip"),
    ]:
        st.markdown(f"<div class='nav-item'>{icon} {name}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='sec'>OCR Languages</div>", unsafe_allow_html=True)
    for l in ["🇮🇳 English","🇮🇳 Tamil · தமிழ்","🇮🇳 Hindi · हिन्दी"]:
        st.markdown(f"<div class='nav-item'>{l}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""
<div class='stat-grid'>
  <div class='stat-box'><div class='stat-n'>{len(st.session_state.processed_docs)}</div><div class='stat-l'>Docs</div></div>
  <div class='stat-box'><div class='stat-n'>{len(st.session_state.chat_history)//2}</div><div class='stat-l'>Questions</div></div>
</div>""", unsafe_allow_html=True)
    if st.button("🗑️ Clear Session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class='topbar'>
  <span style='font-size:2rem'>📄</span>
  <div>
    <p class='t-title'>DocuSense AI</p>
    <p class='t-sub'>Upload any Indian document · AI extraction in English, Tamil &amp; Hindi · Download PDF report · Ask questions</p>
  </div>
</div>""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────
st.markdown("<div class='sec'>Upload Document</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Drop documents here",
    type=["pdf","png","jpg","jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# ── Process new uploads ───────────────────────────────────────
if uploaded_files:
    for file in uploaded_files:
        if file.name in st.session_state.raw_store:
            continue

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"<div class='sec'>⚙️ Processing: {file.name}</div>", unsafe_allow_html=True)

        suffix = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with st.spinner("Enhancing image quality and running multilingual OCR..."):
            raw_text = extract_text(tmp_path)
        os.unlink(tmp_path)

        if not raw_text or raw_text.startswith("OCR Error") or len(raw_text.strip()) < 10:
            st.error(f"❌ Extraction failed: {raw_text}")
            continue

        doc_type  = detect_document_type(raw_text)
        doc_label = get_document_label(doc_type)
        fields    = {}

        with st.spinner("AI extracting structured fields..."):
            try:
                if doc_type in ID_TYPES:
                    result = analyse_id_document(raw_text, doc_type)
                    result.pop("_validation", None)
                    fields = result
                elif doc_type == "financial":
                    fields = {"Summary": extract_financial_summary(raw_text)}
                elif doc_type == "bank_statement":
                    fields = {"Summary": extract_bank_summary(raw_text)}
                elif doc_type == "contract":
                    r = analyse_contract(raw_text)
                    fields = {"Summary": r.get("plain_summary",""),
                              "Top Risks": r.get("top_risks",""),
                              "Missing Clauses": r.get("missing_clauses","")}
                elif doc_type == "invoice":
                    r = analyse_invoice(raw_text)
                    fields = {"Details": r.get("extracted_fields","")}
            except Exception as e:
                fields = {"Error": str(e)}

        # Persist permanently
        st.session_state.raw_store[file.name]      = raw_text
        st.session_state.fields_store[file.name]   = fields
        st.session_state.doctype_store[file.name]  = doc_type
        st.session_state.doclabel_store[file.name] = doc_label
        st.session_state.all_raw_texts.append(f"=== {file.name} ===\n{raw_text}")
        st.session_state.current_raw_text = "\n\n".join(st.session_state.all_raw_texts)
        st.session_state.processed_docs.append({
            "filename": file.name, "doc_type": doc_type, "extracted_fields": fields
        })
        try:
            chain, _ = build_rag_chain(st.session_state.current_raw_text, "session", doc_type)
            st.session_state.rag_chain = chain
        except:
            pass

# ═══════════════════════════════════════════════════════════════
# ALWAYS RENDER stored docs — runs on EVERY rerun
# This keeps extracted text visible even after chatting
# ═══════════════════════════════════════════════════════════════
if st.session_state.raw_store:

    # PDF download button — covers ALL processed docs
    all_docs_data = [
        {
            "filename":  fname,
            "doc_label": st.session_state.doclabel_store.get(fname,"Document"),
            "fields":    st.session_state.fields_store.get(fname,{}),
            "raw_text":  st.session_state.raw_store.get(fname,""),
        }
        for fname in st.session_state.raw_store
    ]
    try:
        pdf_bytes = generate_pdf_report(all_docs_data)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name="DocuSense_Report.pdf",
            mime="application/pdf",
            use_container_width=False,
            key="pdf_main",
        )
    except Exception as e:
        st.warning(f"PDF generation error: {e}")

    # Render each document
    for fname in st.session_state.raw_store:
        raw_text  = st.session_state.raw_store[fname]
        fields    = st.session_state.fields_store.get(fname, {})
        doc_type  = st.session_state.doctype_store.get(fname, "general")
        doc_label = st.session_state.doclabel_store.get(fname, "Document")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"<div class='badge'>🎯 {doc_label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#64748b;margin-bottom:10px'>📎 {fname}</div>",
                    unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        # Left — fields
        with col_left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec'>Extracted Fields</div>", unsafe_allow_html=True)
            display = {k: v for k, v in fields.items()
                       if v and str(v) not in ("null","None","") and not k.startswith("_")}
            if display:
                rows = "<table class='ftable'>"
                for f_, v_ in display.items():
                    rows += f"<tr><td class='fl'>{f_}</td><td class='fv'>{str(v_)}</td></tr>"
                rows += "</table>"
                st.markdown(rows, unsafe_allow_html=True)
            else:
                st.info("See extracted text panel →")
            st.markdown("</div>", unsafe_allow_html=True)

        # Right — raw text (ALWAYS visible)
        with col_right:
            safe_text = raw_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(f"""
<div class='raw-hdr'>
  <span class='raw-title'>Extracted Text</span>
  <span class='raw-ok'>✅ {len(raw_text)} chars</span>
</div>
<div class='raw-box'>{safe_text}</div>""", unsafe_allow_html=True)

    # Cross-document verification
    id_docs = [d for d in st.session_state.processed_docs
               if d["doc_type"] in ["aadhaar","pan","voter_id","passport","driving_licence"]]
    if len(id_docs) >= 2:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='sec'>Cross-Document Verification</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:13px;color:#334155'>Compare name, DOB &amp; address across uploaded ID documents.</span>",
                    unsafe_allow_html=True)
        if st.button("🔍 Verify Consistency", type="primary", use_container_width=True):
            with st.spinner("Comparing..."):
                result = cross_verify_documents(id_docs)
                st.markdown(result["verification_report"])
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty state
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1,"🪪","ID Documents","Aadhaar · PAN · Voter ID · Passport · DL"),
        (c2,"📊","Financial Docs","Bank statements · P&L · Invoices · Salary slips"),
        (c3,"📝","Legal Docs","Contracts · NDAs · Agreements"),
    ]:
        with col:
            st.markdown(f"""
<div class='card' style='text-align:center;padding:1.4rem 1rem;'>
  <div style='font-size:2rem;margin-bottom:8px'>{icon}</div>
  <div style='font-size:14px!important;font-weight:600;color:#0f172a!important;margin-bottom:4px'>{title}</div>
  <div style='font-size:13px!important;color:#64748b!important;line-height:1.5'>{desc}</div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CHATBOT — below extracted text, always visible
# ═══════════════════════════════════════════════════════════════
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='margin-bottom:10px'>
  <span style='font-size:15px!important;font-weight:600;color:#0f172a!important'>💬 Ask anything about your documents</span><br>
  <span style='font-size:13px!important;color:#64748b!important'>Short, direct answers — grounded in your extracted text above</span>
</div>""", unsafe_allow_html=True)

if not st.session_state.rag_chain:
    st.markdown("""
<div class='card'>
  <div class='chat-hint'>⬆️ Upload a document above to start asking questions</div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown("""<div class='chat-hint'>
💡 <b>"What is the Aadhaar number?"</b><br>
<b>"What is the PAN number?"</b><br>
<b>"What is the full address?"</b><br>
<b>"What is the name in Tamil?"</b>
</div>""", unsafe_allow_html=True)
    else:
        html = "<div>"
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                html += f"<div class='msg-user'>🧑 {msg['content']}</div>"
            else:
                content = msg["content"].replace('\n','<br>')
                html += f"<div class='msg-bot'>🤖 {content}</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if question := st.chat_input("Ask about your document..."):
        st.session_state.chat_history.append({"role":"user","content":question})
        with st.spinner("Searching..."):
            try:
                from groq import Groq
                gc   = Groq(api_key=os.getenv("GROQ_API_KEY"))
                resp = gc.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a document analyst. Rules:
1. Answer ONLY from the document text — nothing else
2. Be SHORT and DIRECT — give only the exact value asked
3. Do NOT explain or add context
4. If asked for a number → just the number
5. If asked for a name → just the name
6. Search English, Tamil, and Hindi sections
7. Not found → say: Not found in the document.

Examples:
Q: Aadhaar number? → 4132 8515 7430
Q: Name? → Harshitha V
Q: Address? → No 278/1, Vivekananda Nagar 3rd Street, Singanallur, Coimbatore 641005"""
                        },
                        {
                            "role": "user",
                            "content": f"Document:\n{st.session_state.current_raw_text}\n\nQuestion: {question}"
                        }
                    ],
                    temperature=0, max_tokens=150,
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                try:
                    answer = st.session_state.rag_chain.invoke(question)
                except:
                    answer = f"Error: {e}"
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        st.rerun()
