"""
AI Document Analysis REST API
==============================
Production-grade FastAPI application that accepts documents via REST API,
extracts text, and uses Google Gemini AI to return structured analysis.

Deployment Notes:
- For Railway: add PORT environment variable, use:
    uvicorn src.main:app --host 0.0.0.0 --port $PORT
- For Render: set start command to:
    uvicorn src.main:app --host 0.0.0.0 --port 10000
- Tesseract binary must be installed on the server.
    For Railway, add a nixpacks.toml with:
        [phases.setup]
        aptPkgs = ["tesseract-ocr"]
    For Render, add to your build command:
        apt-get update && apt-get install -y tesseract-ocr
"""

import os
import io
import json
import base64
import logging
from openai import OpenAI

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pdfplumber
from docx import Document
from PIL import Image
import pytesseract

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "sk_track2_987654321")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_nvidia_client():
    key = os.getenv("NVIDIA_API_KEY", "")
    if key:
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key,
        )
    return None

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Document Analysis API",
    description="Accepts PDF, DOCX, and image files as base64, extracts text, "
                "and uses Google Gemini AI for summarisation, entity extraction, and "
                "sentiment analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class DocumentRequest(BaseModel):
    fileName: str
    fileType: str
    fileBase64: str


class EntitiesModel(BaseModel):
    names: list[str] = []
    dates: list[str] = []
    organizations: list[str] = []
    amounts: list[str] = []
    locations: list[str] = []


class FinancialDetails(BaseModel):
    total_amount: str = ""
    currency: str = ""
    tax: str = ""
    due_date: str = ""


class ContactDetails(BaseModel):
    email: str = ""
    phone: str = ""


class SuccessResponse(BaseModel):
    status: str = "success"
    fileName: str
    document_type: str = "unknown"
    summary: str
    key_points: list[str] = []
    entities: EntitiesModel
    contact_details: ContactDetails = ContactDetails()
    financial_details: FinancialDetails = FinancialDetails()
    payment_status: str = "Unknown"
    sentiment: str
    confidence: float = 0.0


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


# ---------------------------------------------------------------------------
# Text Extraction Helpers
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_parts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.warning("No text extracted from PDF page %s", page.page_number)
    except Exception as exc:
        logger.exception("PDF extraction error:")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {exc}")
    
    if not text_parts:
        return ""
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    text_parts: list[str] = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
    except Exception as exc:
        logger.error("DOCX extraction error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to extract text from DOCX: {exc}")
    return "\n".join(text_parts)


def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from image bytes using pytesseract OCR."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        logger.error("Image OCR error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to extract text from image: {exc}")
    return text.strip()


EXTRACTORS = {
    "pdf": extract_text_from_pdf,
    "docx": extract_text_from_docx,
    "image": extract_text_from_image,
}

# ---------------------------------------------------------------------------
# Gemini AI Analysis
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are an advanced AI Document Intelligence System.

Your task is to analyze structured or unstructured content extracted from PDFs or images (OCR text).

STRICT RULES (MANDATORY):
- Return ONLY a valid JSON object.
- Do NOT include any explanation or extra text.
- Do NOT include markdown (no ```json).
- Output must start with { and end with }.
- Ensure no missing or extra braces.
- Use double quotes for all keys and values.
- Do NOT include trailing commas.

OUTPUT FORMAT:
{
  "status": "success",
  "document_type": "invoice | receipt | report | letter | unknown",
  "summary": "Write a professional 2-3 sentence summary clearly mentioning who issued the document, who received it, what service/product was involved, amount, and date.",
  "key_points": ["Key insight 1", "Key insight 2", "Key insight 3", "Key insight 4"],
  "entities": {
    "names": [],
    "dates": [],
    "organizations": [],
    "amounts": [],
    "locations": []
  },
  "contact_details": {
    "email": "",
    "phone": ""
  },
  "financial_details": {
    "total_amount": "",
    "currency": "INR | USD | etc",
    "tax": "",
    "due_date": ""
  },
  "payment_status": "Completed | Pending | Unknown",
  "sentiment": "Positive | Neutral | Negative",
  "confidence": 0.0
}

INTELLIGENT EXTRACTION RULES:
- Identify who issued the document and who received it
- Extract email and phone if present
- Detect payment status (Completed / Pending)
- Infer currency from symbols (₹ → INR, $ → USD)
- Extract location if available
- If it is an invoice, prioritize financial and service-related details
- If any field is missing, return "" or []

SUMMARY REQUIREMENTS:
- Must be clear, professional, and informative
- Include: person/company, service, amount, and date
- Avoid vague or generic sentences

FAIL-SAFE:
If you cannot strictly follow the format, return:
{"status":"error","message":"invalid_json"}

INPUT:
{text}"""


def fallback_analysis(extracted_text: str) -> dict:
    """Return a safe fallback response when AI is unavailable."""
    return {
        "document_type": "unknown",
        "summary": extracted_text[:200].strip(),
        "key_points": [],
        "entities": {"names": [], "dates": [], "organizations": [], "amounts": [], "locations": []},
        "contact_details": {"email": "", "phone": ""},
        "financial_details": {"total_amount": "", "currency": "", "tax": "", "due_date": ""},
        "payment_status": "Unknown",
        "sentiment": "Neutral",
        "confidence": 0.0,
    }


def _parse_response(raw: str) -> dict:
    """Extract and parse JSON from model response."""
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```")).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    result = json.loads(raw)
    if result.get("status") == "error":
        raise ValueError("Model returned error status")
    if not all(k in result for k in ("summary", "entities", "sentiment")):
        raise ValueError("Response missing required fields")
    if result["sentiment"] not in {"Positive", "Neutral", "Negative"}:
        result["sentiment"] = "Neutral"
    if result.get("payment_status") not in {"Completed", "Pending", "Unknown"}:
        result["payment_status"] = "Unknown"
    entities = result.get("entities", {})
    for key in ("names", "dates", "organizations", "amounts", "locations"):
        if key not in entities or not isinstance(entities[key], list):
            entities[key] = []
    result["entities"] = entities
    fin = result.get("financial_details", {})
    for key in ("total_amount", "currency", "tax", "due_date"):
        if key not in fin:
            fin[key] = ""
    result["financial_details"] = fin
    contact = result.get("contact_details", {})
    result["contact_details"] = {"email": contact.get("email", ""), "phone": contact.get("phone", "")}
    result.setdefault("document_type", "unknown")
    result.setdefault("key_points", [])
    result.setdefault("confidence", 0.0)
    return result


def analyse_with_gemini(extracted_text: str) -> dict:
    """Single NVIDIA API call with one retry. Falls back gracefully on errors."""
    client = get_nvidia_client()
    if not client:
        logger.warning("NVIDIA client not configured — using fallback")
        return fallback_analysis(extracted_text)

    prompt = ANALYSIS_PROMPT.replace("{text}", extracted_text)
    raw = ""

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            return _parse_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Parse error (attempt %d): %s | raw: %s", attempt + 1, exc, raw[:300])
            if attempt == 1:
                return fallback_analysis(extracted_text)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate" in exc_str.lower():
                logger.warning("NVIDIA 429 rate limit — using fallback")
                return fallback_analysis(extracted_text)
            if "401" in exc_str or "unauthorized" in exc_str.lower():
                logger.error("NVIDIA 401 invalid API key — using fallback")
                return fallback_analysis(extracted_text)
            logger.error("NVIDIA API error (attempt %d): %s", attempt + 1, exc)
            if attempt == 1:
                return fallback_analysis(extracted_text)

    return fallback_analysis(extracted_text)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "running"}


@app.post("/api/document-analyze")
async def document_analyze(request: Request, body: DocumentRequest):
    """
    Accept a document as base64, extract text, analyse with Gemini AI,
    and return structured results.
    """

    # --- 1. Validate API key ---
    api_key = request.headers.get("x-api-key", "")
    if not api_key or api_key != API_SECRET_KEY:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Unauthorized"},
        )

    # --- 2. Decode base64 ---
    try:
        file_bytes = base64.b64decode(body.fileBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding in fileBase64")

    # --- 3. Extract text based on fileType ---
    file_type = body.fileType.lower()
    extractor = EXTRACTORS.get(file_type)
    if extractor is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported fileType '{body.fileType}'. Supported: pdf, docx, image",
        )
    extracted_text = extractor(file_bytes)

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the document")

    # --- 4. Analyse with Gemini (falls back gracefully on quota/errors) ---
    analysis = analyse_with_gemini(extracted_text)

    # --- 5. Build and return response ---
    return JSONResponse(content=json.loads(SuccessResponse(
        fileName=body.fileName,
        document_type=analysis.get("document_type", "unknown"),
        summary=analysis["summary"],
        key_points=analysis.get("key_points", []),
        entities=EntitiesModel(**analysis["entities"]),
        contact_details=ContactDetails(**analysis.get("contact_details", {})),
        financial_details=FinancialDetails(**analysis.get("financial_details", {})),
        payment_status=analysis.get("payment_status", "Unknown"),
        sentiment=analysis["sentiment"],
        confidence=analysis.get("confidence", 0.0),
    ).model_dump_json()), media_type="application/json")


# ---------------------------------------------------------------------------
# Entry-point (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
