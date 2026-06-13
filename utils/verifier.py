import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def cross_verify_documents(doc_data_list: list) -> dict:
    """
    Compare multiple extracted documents for consistency.
    doc_data_list: [{"doc_type": "aadhaar", "extracted_fields": {...}}, ...]
    """
    if len(doc_data_list) < 2:
        return {"verification_report": "Need at least 2 documents to cross-verify."}
    
    summary = "\n\n".join([
        f"Document Type: {d['doc_type'].replace('_', ' ').title()}\n"
        f"Extracted Fields: {d['extracted_fields']}"
        for d in doc_data_list
    ])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""
You are a document verification expert. Compare these documents and check for inconsistencies.

Look for:
1. Name spelling differences across documents
2. Date of birth mismatches
3. Address differences
4. Any suspicious patterns or red flags
5. Fields that should match but don't

Documents:
{summary}

Provide:
- MATCHES: fields that are consistent across documents
- MISMATCHES: fields that differ (with exact values)
- VERDICT: VERIFIED / NEEDS REVIEW / SUSPICIOUS
- RECOMMENDATION: what action to take
"""
        }],
        temperature=0,
        max_tokens=1500,
    )
    
    return {
        "verification_report": response.choices[0].message.content,
        "documents_compared": len(doc_data_list),
    }

def check_document_expiry(extracted_fields: dict, doc_type: str) -> dict:
    """Check if document is expired."""
    from datetime import datetime
    
    expiry_keys = {
        "passport":        "Date of Expiry",
        "driving_licence": "Validity Date (Non-Transport)",
    }
    
    key = expiry_keys.get(doc_type)
    if not key:
        return {"status": "N/A", "message": "Expiry check not applicable for this document type"}
    
    expiry_str = extracted_fields.get(key, "")
    if not expiry_str:
        return {"status": "UNKNOWN", "message": "Expiry date not found in document"}
    
    return {
        "status": "CHECK_MANUALLY",
        "expiry_date": expiry_str,
        "message": f"Please verify expiry date: {expiry_str}"
    }
