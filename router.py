import re

DOCUMENT_TYPES = {
    "aadhaar":          ["aadhaar", "uid", "unique identification", "uidai", "enrolment no"],
    "pan":              ["permanent account number", "income tax", "pan card", "pan no"],
    "voter_id":         ["election commission", "epic", "voter", "electoral photo"],
    "passport":         ["passport", "republic of india", "nationality", "place of birth"],
    "driving_licence":  ["driving licence", "motor vehicles", "transport authority", "dl no"],
    "bank_statement":   ["account statement", "opening balance", "closing balance", "ifsc", "transaction date"],
    "financial":        ["profit and loss", "balance sheet", "revenue", "ebitda", "net profit", "total assets", "liabilities"],
    "contract":         ["agreement", "whereas", "hereinafter", "terms and conditions", "indemnify", "arbitration", "party of the first"],
    "invoice":          ["invoice", "bill to", "ship to", "gst", "hsn", "total amount due", "payable within"],
    "salary_slip":      ["salary slip", "basic pay", "hra", "pf deduction", "net salary", "payslip", "employee id"],
}

def detect_document_type(text: str) -> str:
    """Auto-detect document type from extracted text."""
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in DOCUMENT_TYPES.items():
        scores[doc_type] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

def get_document_label(doc_type: str) -> str:
    labels = {
        "aadhaar":         "Aadhaar Card",
        "pan":             "PAN Card",
        "voter_id":        "Voter ID",
        "passport":        "Passport",
        "driving_licence": "Driving Licence",
        "bank_statement":  "Bank Statement",
        "financial":       "Financial Statement",
        "contract":        "Contract / Agreement",
        "invoice":         "Invoice / Bill",
        "salary_slip":     "Salary Slip",
        "general":         "General Document",
    }
    return labels.get(doc_type, "Unknown Document")
