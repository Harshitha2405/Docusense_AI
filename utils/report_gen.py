from fpdf import FPDF
from datetime import datetime
import re

class Report(FPDF):
    def header(self):
        self.set_fill_color(37, 99, 235)
        self.rect(0, 0, 210, 16, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.set_xy(0, 3)
        self.cell(0, 10, "DocuSense AI  -  Document Intelligence Report", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8,
            f"DocuSense AI  |  {datetime.now().strftime('%d %b %Y %H:%M')}  |  Page {self.page_no()}",
            align="C")

def safe(text: str) -> str:
    out = ""
    for ch in str(text):
        cp = ord(ch)
        if cp < 128:
            out += ch
        elif 0x0B80 <= cp <= 0x0BFF:
            out += "[Tamil]"
        elif 0x0900 <= cp <= 0x097F:
            out += "[Hindi]"
        else:
            out += "?"
    return out

def generate_pdf_report(docs_data: list) -> bytes:
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 22, 16)

    # ── Cover page ────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(4)
    pdf.cell(0, 10, "Document Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7,
        f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}  |  "
        f"{len(docs_data)} document(s)",
        ln=True, align="C")
    pdf.ln(8)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.5)
    pdf.line(16, pdf.get_y(), 194, pdf.get_y())
    pdf.ln(6)

    # Summary table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(37, 99, 235)
    pdf.set_text_color(255, 255, 255)
    col_w = [10, 62, 38, 68]
    for txt, w in zip(["#", "File Name", "Document Type", "Key Information"], col_w):
        pdf.cell(w, 9, txt, fill=True, align="C" if txt == "#" else "L")
    pdf.ln()

    for i, doc in enumerate(docs_data):
        pdf.set_fill_color(248, 250, 252) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "", 9)
        fields   = doc.get("fields", {})
        key_info = ""
        for key in ["Aadhaar Number","PAN Number","Name (English)","Name",
                    "Employee Name","DL Number","EPIC Number","Passport Number"]:
            if fields.get(key):
                key_info = f"{key}: {str(fields[key])[:35]}"
                break
        if not key_info and fields:
            fk = next((k for k in fields if not k.startswith("_")), None)
            if fk:
                key_info = f"{fk}: {str(fields[fk])[:35]}"
        fname = doc.get("filename","")
        if len(fname) > 36: fname = fname[:33] + "..."
        pdf.cell(col_w[0], 8, str(i+1),                       fill=True, align="C")
        pdf.cell(col_w[1], 8, safe(fname),                     fill=True)
        pdf.cell(col_w[2], 8, safe(doc.get("doc_label","")),  fill=True)
        pdf.cell(col_w[3], 8, safe(key_info),                  fill=True)
        pdf.ln()

    pdf.ln(8)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(16, pdf.get_y(), 194, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, "Detailed extraction follows on the next pages.", ln=True)

    # ── One page per document ─────────────────────────────
    for doc in docs_data:
        pdf.add_page()
        fname     = safe(doc.get("filename", ""))
        doc_label = safe(doc.get("doc_label", "Document"))
        fields    = doc.get("fields", {})
        raw_text  = doc.get("raw_text", "")

        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 9, doc_label, ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"File: {fname}", ln=True)
        pdf.ln(2)
        pdf.set_draw_color(226, 232, 240)
        pdf.set_line_width(0.4)
        pdf.line(16, pdf.get_y(), 194, pdf.get_y())
        pdf.ln(5)

        display = {k: v for k, v in fields.items()
                   if v and str(v) not in ("null","None","") and not k.startswith("_")}
        if display:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 6, "EXTRACTED FIELDS", ln=True)
            pdf.ln(1)
            for j, (field, value) in enumerate(display.items()):
                pdf.set_fill_color(248, 250, 252) if j % 2 == 0 else pdf.set_fill_color(255, 255, 255)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(72, 9, safe(str(field)), border="B", fill=True)
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(15, 23, 42)
                val = safe(str(value))
                if len(val) > 60: val = val[:57] + "..."
                pdf.cell(0, 9, val, border="B", fill=True, ln=True)
            pdf.ln(6)

        if raw_text.strip():
            pdf.set_draw_color(226, 232, 240)
            pdf.line(16, pdf.get_y(), 194, pdf.get_y())
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(100, 116, 139)
            pdf.cell(0, 6, "EXTRACTED TEXT (OCR)", ln=True)
            pdf.ln(1)
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(30, 41, 59)
            count = 0
            for line in raw_text.split('\n'):
                if count > 90: break
                line = line.strip()
                if not line: continue
                try:
                    pdf.multi_cell(0, 5, safe(line))
                    count += 1
                except:
                    pass

    return bytes(pdf.output())
