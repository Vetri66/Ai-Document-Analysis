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

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini client
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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

SYSTEM_PROMPT = """You are a document analysis AI. Analyse the provided text and respond ONLY with a raw JSON object. No markdown, no backticks, no explanation outside the JSON.

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
- Return ONLY the JSON object, nothing else"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception), # In production, refine this to specific API errors
    before_sleep=lambda retry_state: logger.warning(f"Retrying Gemini API... Attempt {retry_state.attempt_number}")
)
def analyse_with_gemini(extracted_text: str) -> dict:
    """Send extracted text to Gemini and return parsed JSON analysis."""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Analyse the following document text:\n\n{extracted_text}",
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.1,
            },
        )
    except Exception as exc:
        logger.exception("Gemini API error detail:")
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}")

    raw_response = response.text.strip()

    # Strip markdown code fences if Gemini wraps the JSON
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_response = "\n".join(lines).strip()

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        logger.error("Failed to parse Gemini response as JSON: %s", raw_response)
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid JSON response. Please try again.",
        )

    # Validate required keys
    if "summary" not in result or "entities" not in result or "sentiment" not in result:
        raise HTTPException(
            status_code=502,
            detail="Gemini response missing required fields.",
        )

    # Ensure sentiment is one of the allowed values
    allowed_sentiments = {"Positive", "Neutral", "Negative"}
    if result["sentiment"] not in allowed_sentiments:
        result["sentiment"] = "Neutral"

    # Ensure all entity sub-keys exist
    entities = result.get("entities", {})
    for key in ("names", "dates", "organizations", "amounts"):
        if key not in entities or not isinstance(entities[key], list):
            entities[key] = []
    result["entities"] = entities

    return result


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

    # --- 4. Send to Gemini AI ---
    analysis = analyse_with_gemini(extracted_text)

    # --- 5. Build and return response ---
    return SuccessResponse(
        status="success",
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
