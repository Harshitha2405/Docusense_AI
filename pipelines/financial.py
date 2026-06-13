import os
from dotenv import load_dotenv

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older environments
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

FINANCIAL_PROMPT = PromptTemplate.from_template("""
You are a senior financial analyst. Use ONLY the context below to answer the question.
If the answer is not in the context, say "This information is not available in the document."

Context:
{context}

Question: {question}

Provide:
1. Direct answer to the question
2. Key numbers or metrics supporting your answer
3. Any financial red flags or risks you notice
""")

GENERAL_RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful document analyst. Use ONLY the context below to answer.
If not found, say "Not found in the document."

Context:
{context}

Question: {question}

Answer clearly and concisely:
""")

def build_rag_chain(text: str, session_id: str, doc_type: str = "general"):
    """Build a RAG chain for any document type."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.split_documents([Document(page_content=text, metadata={"source": session_id})])
    
    persist_dir = f"chroma_db/{session_id}"
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    prompt = FINANCIAL_PROMPT if doc_type in ["financial", "bank_statement"] else GENERAL_RAG_PROMPT
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def extract_financial_summary(text: str) -> str:
    """Generate a quick financial summary using LLM."""
    response = llm.invoke(f"""
You are a financial analyst. From the document below, extract and summarise:

1. Revenue / Turnover (with figures)
2. Net Profit / Loss (with figures)
3. Total Assets
4. Total Liabilities
5. Key financial ratios (if present)
6. Top 3 financial risks or red flags

Document:
{text[:4000]}

Format your response clearly with headings.
""")
    return response.content

def extract_bank_summary(text: str) -> str:
    """Summarise a bank statement."""
    response = llm.invoke(f"""
Analyse this bank statement and provide:

1. Account holder name and account number (masked)
2. Statement period
3. Opening and closing balance
4. Total credits and debits
5. Largest transactions
6. Spending patterns observed
7. Any suspicious activity

Bank Statement:
{text[:4000]}

Be specific with numbers found in the document.
""")
    return response.content
