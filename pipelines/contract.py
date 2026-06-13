import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

RISK_CATEGORIES = [
    "Termination clauses",
    "Penalty and liability clauses",
    "Non-compete and non-solicitation",
    "Intellectual property ownership",
    "Dispute resolution and arbitration",
    "Confidentiality obligations",
    "Payment terms and conditions",
    "Governing law and jurisdiction",
    "Force majeure",
    "Indemnification",
]

STANDARD_CLAUSES = [
    "Force majeure",
    "Limitation of liability",
    "Indemnification",
    "Data protection / GDPR",
    "Exit / termination clause",
    "Dispute resolution",
    "Governing law",
    "Entire agreement",
    "Severability",
    "Amendment procedure",
]

def analyse_contract(text: str) -> dict:
    """Full contract analysis: risk scoring, missing clauses, top risks."""
    
    # 1. Risk scoring per clause category
    risk_response = llm.invoke(f"""
Analyse this contract. For each category, rate the risk as HIGH / MEDIUM / LOW and give ONE sentence reason.

Categories:
{chr(10).join(f"- {c}" for c in RISK_CATEGORIES)}

Contract:
{text[:4000]}

Format exactly as:
Category | Risk Level | Reason
(one line per category)
""")

    # 2. Top 3 risks summary
    summary_response = llm.invoke(f"""
What are the TOP 3 risks in this contract for the signing party?
Be specific — reference actual clauses if possible.

Contract:
{text[:3000]}

Format as 3 bullet points.
""")

    # 3. Missing standard clauses
    missing_response = llm.invoke(f"""
Check which of these standard clauses are MISSING from the contract:
{chr(10).join(f"- {c}" for c in STANDARD_CLAUSES)}

Contract:
{text[:3000]}

List only the missing ones. If all present, say "All standard clauses are present."
""")

    # 4. Plain English summary
    plain_response = llm.invoke(f"""
Summarise this contract in plain English in 5 bullet points.
Avoid legal jargon. Write as if explaining to someone with no legal background.

Contract:
{text[:3000]}
""")

    return {
        "risk_analysis":   risk_response.content,
        "top_risks":       summary_response.content,
        "missing_clauses": missing_response.content,
        "plain_summary":   plain_response.content,
    }

def analyse_invoice(text: str) -> dict:
    """Extract and validate invoice fields."""
    response = llm.invoke(f"""
Extract these fields from the invoice:
- Invoice Number
- Invoice Date
- Due Date
- Vendor Name
- Vendor GST Number
- Buyer Name
- Buyer GST Number
- Line Items (item, quantity, rate, amount)
- Subtotal
- GST Amount
- Total Amount Due
- Payment Terms

Invoice:
{text}

Return as structured text with clear labels.
""")
    
    # GST validation
    gst_pattern = r'[0-9]{{2}}[A-Z]{{5}}[0-9]{{4}}[A-Z][1-9A-Z]Z[0-9A-Z]'
    gst_numbers = __import__('re').findall(gst_pattern, text)
    
    return {
        "extracted_fields": response.content,
        "gst_numbers_found": gst_numbers,
        "gst_valid": len(gst_numbers) > 0,
    }
