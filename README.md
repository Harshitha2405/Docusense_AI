# DocuSense AI

DocuSense AI is a multilingual document intelligence platform that extracts, analyzes, verifies, and answers questions about Indian documents using OCR, RAG, and LLM-based field extraction.

It supports Aadhaar, PAN, Voter ID, Passport, Driving Licence, Bank Statements, Financial Statements, Contracts, Invoices, and Salary Slips across English, Tamil, and Hindi.

## What It Does

- Detects document type automatically from OCR text.
- Extracts structured fields from government, financial, and legal documents.
- Enhances scanned images before OCR using OpenCV preprocessing.
- Answers questions grounded in the uploaded document content.
- Generates downloadable PDF reports for uploaded files.
- Verifies consistency across multiple identity documents.
- Stores uploaded document data in session state for repeated analysis.

## Recent Improvements

- Added dynamic Tesseract discovery for Windows and PATH-based setups.
- Updated embeddings usage to support the newer `langchain_huggingface` package.
- Added smoke tests for routing and PDF generation.
- Fixed PDF rendering compatibility with built-in FPDF fonts.
- Improved startup resilience for missing OCR dependencies.

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Groq LLaMA 3
- Tesseract OCR
- OpenCV
- HuggingFace sentence-transformers
- FPDF2

## Project Structure

```text
docusense_fixed/
├── app.py                  Streamlit UI and workflow
├── router.py               Document type routing
├── ocr/
│   └── extractor.py        OCR and image preprocessing
├── pipelines/
│   ├── id_card.py          Aadhaar, PAN, Passport, DL, Voter ID
│   ├── financial.py        Financial extraction and RAG chain
│   └── contract.py         Contract and invoice analysis
├── utils/
│   ├── report_gen.py       PDF report generation
│   └── verifier.py         Cross-document verification
├── tests/
│   └── test_smoke.py       Basic smoke tests
├── chroma_db/              Local vector store
├── requirements.txt        Python dependencies
└── README.md               Project documentation
```

## Setup

### 1. Install Tesseract OCR

Windows: download it from the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) and install it.

Mac:

```bash
brew install tesseract
```

Ubuntu:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

If Tesseract is not on your PATH, set one of these environment variables:

- `TESSERACT_CMD`
- `TESSERACT_PATH`

Example on Windows:

```bash
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Groq

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Run tests

```bash
python -m unittest discover -s tests -v
```

## Usage

1. Upload one or more documents from the Streamlit UI.
2. Let the app detect the document type and extract fields.
3. Use the Q&A box to ask document-specific questions.
4. Download the generated PDF report.
5. Upload multiple identity documents to run cross-document verification.

## Notes

- The app uses Groq for extraction and question answering, so an API key is required.
- The OCR pipeline is optimized for scanned documents, but quality still depends on the source image.
- The local ChromaDB folder is created automatically when the RAG pipeline runs.

## Validation

The following checks pass in the current workspace:

- `pip install -r requirements.txt`
- `python -m unittest discover -s tests -v`

## License

No license has been added yet. Add one before publishing publicly on GitHub if needed.
