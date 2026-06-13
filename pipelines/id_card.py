import re
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPTS = {
    "aadhaar": """You are extracting data from an Aadhaar card. The OCR text contains English, Tamil, and Hindi versions of the same content.

Extract these fields carefully:
- Name (English) — look for Latin script name e.g. "Harshitha V"
- Name (Tamil) — look for Tamil script name e.g. "ஹர்ஷிதா வி"
- Date of Birth — format DD/MM/YYYY, look near "DOB" or "Date of Birth"
- Gender — Male or Female
- Aadhaar Number — CRITICAL: look for pattern "XXXX XXXX XXXX" near "Your Aadhaar No." or "உங்கள் ஆதார் எண்" — it is always 12 digits in groups of 4
- Father/Mother/Husband Name — look near "S/O", "D/O", "W/O"
- Address — full address with door no, street, city, pincode
- Mobile Number — 10 digit number near "Mobile:"
- Enrollment Number — near "Enrolment No."

Return ONLY this exact JSON:
{
  "Name (English)": "",
  "Name (Tamil)": "",
  "Date of Birth": "",
  "Gender": "",
  "Aadhaar Number": "",
  "Father/Guardian Name": "",
  "Address": "",
  "Mobile Number": "",
  "Enrollment Number": ""
}
Use null for missing fields.""",

    "pan": """Extract from this PAN card OCR text:
- Name
- Father Name
- Date of Birth (DD/MM/YYYY)
- PAN Number (5 letters + 4 digits + 1 letter e.g. ABCDE1234F)

Return ONLY valid JSON.""",

    "voter_id": """Extract from this Voter ID OCR text:
- Name (English)
- Name (regional script if present)
- Father/Husband Name
- Date of Birth
- EPIC Number
- Address
- Assembly Constituency
Return ONLY valid JSON.""",

    "passport": """Extract from this Passport OCR text:
- Full Name, Date of Birth, Nationality, Passport Number
- Date of Issue, Date of Expiry, Place of Birth, Gender
Return ONLY valid JSON.""",

    "driving_licence": """Extract from this Driving Licence OCR text:
- Name, Date of Birth, DL Number
- Date of Issue, Validity, Vehicle Classes, Address, Blood Group
Return ONLY valid JSON.""",

    "salary_slip": """Extract from this Salary Slip:
- Employee Name, Employee ID, Designation, Month Year
- Basic Pay, HRA, Gross Salary, Net Salary, PF Deduction
Return ONLY valid JSON.""",
}

def analyse_id_document(text: str, doc_type: str) -> dict:
    prompt = EXTRACTION_PROMPTS.get(doc_type, "Extract all key fields as JSON.")

    # Pre-extract Aadhaar number using regex as reliable backup
    aadhaar_matches = re.findall(r'\b(\d{4}\s\d{4}\s\d{4})\b', text)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You extract structured data from OCR text of Indian government documents.
The text has English, Tamil, and Hindi mixed together — use ALL sections to find the best data.
Return ONLY raw valid JSON. No markdown, no explanation, no extra text."""
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nOCR TEXT:\n{text}"
            }
        ],
        temperature=0,
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'```json|```', '', raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {"raw_output": raw}

    # Inject Aadhaar number from regex if LLM missed it
    if doc_type == "aadhaar" and aadhaar_matches:
        current = result.get("Aadhaar Number", "") or ""
        if not re.search(r'\d{4}\s\d{4}\s\d{4}', str(current)):
            result["Aadhaar Number"] = aadhaar_matches[0]

    result["_validation"] = validate_id(result, doc_type)
    return result

def validate_id(data: dict, doc_type: str) -> list:
    issues = []
    if doc_type == "aadhaar":
        num = str(data.get("Aadhaar Number", "") or "")
        digits = re.sub(r'\D', '', num)
        if len(digits) == 12:
            data["Aadhaar Number"] = f"{digits[:4]} {digits[4:8]} {digits[8:]}"
            issues.append(("success", "Aadhaar number verified — 12 digits"))
        else:
            issues.append(("error", f"Aadhaar number incomplete ({len(digits)} digits found)"))
    elif doc_type == "pan":
        pan = str(data.get("PAN Number", "") or "").upper()
        if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan):
            issues.append(("success", "PAN number format valid"))
        else:
            issues.append(("warning", f"PAN number may be incorrect: {pan}"))
    if not issues:
        issues.append(("success", "Extraction complete"))
    return issues
