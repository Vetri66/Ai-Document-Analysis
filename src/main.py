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


class SuccessResponse(BaseModel):
    status: str = "success"
    fileName: str
    summary: str
    entities: EntitiesModel
    sentiment: str


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

GEMINI_PROMPT = """You are a document analysis AI. Analyse the provided text and respond ONLY with a raw JSON object. No markdown, no backticks, no explanation outside the JSON.

Required JSON structure:
{
  "summary": "2-4 sentence concise and accurate summary of the document",
  "entities": {
    "names": ["list of person names found"],
    "dates": ["list of dates found"],
    "organizations": ["list of organization names found"],
    "amounts": ["list of monetary amounts found"]
  },
  "sentiment": "Positive | Neutral | Negative"
}

Rules:
- All arrays may be empty [] if nothing found
- sentiment must be exactly one of: Positive, Neutral, Negative
- Return ONLY the JSON object, nothing else

Document text:
{text}"""


def fallback_analysis(extracted_text: str) -> dict:
    """Return a safe fallback response when Gemini is unavailable."""
    return {
        "summary": extracted_text[:200].strip(),
        "entities": {"names": [], "dates": [], "organizations": [], "amounts": []},
        "sentiment": "Neutral",
    }


def _parse_gemini_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini response."""
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    result = json.loads(raw)
    if not all(k in result for k in ("summary", "entities", "sentiment")):
        raise ValueError("Gemini response missing required fields")
    if result["sentiment"] not in {"Positive", "Neutral", "Negative"}:
        result["sentiment"] = "Neutral"
    entities = result.get("entities", {})
    for key in ("names", "dates", "organizations", "amounts"):
        if key not in entities or not isinstance(entities[key], list):
            entities[key] = []
    result["entities"] = entities
    return result


def analyse_with_gemini(extracted_text: str) -> dict:
    """Single NVIDIA API call with one retry. Falls back gracefully on errors."""
    client = get_nvidia_client()
    if not client:
        logger.warning("NVIDIA client not configured — using fallback")
        return fallback_analysis(extracted_text)

    prompt = GEMINI_PROMPT.replace("{text}", extracted_text)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            # Fix truncated JSON by ensuring it ends with closing braces
            if not raw.endswith("}"):
                raw = raw.rstrip() + "\n}\n}"
            return _parse_gemini_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Parse error (attempt %d): %s", attempt + 1, exc)
            logger.error("Raw response was: %s", raw if 'raw' in dir() else 'N/A')
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
    return SuccessResponse(
        fileName=body.fileName,
        summary=analysis["summary"],
        entities=EntitiesModel(**analysis["entities"]),
        sentiment=analysis["sentiment"],
    )


# ---------------------------------------------------------------------------
# Entry-point (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
