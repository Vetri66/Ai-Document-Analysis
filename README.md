---
title: AI Document Analysis API
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# AI Document Analysis API

## Description
A production-ready REST API that accepts PDF, DOCX, and image files as base64,
extracts text, and uses **NVIDIA AI (LLaMA 3.1)** for intelligent document analysis —
including summarisation, entity extraction, financial details, contact info, and sentiment analysis.

## Tech Stack
- Python + FastAPI
- pdfplumber (PDF extraction)
- python-docx (DOCX extraction)
- pytesseract + Pillow (OCR for images)
- NVIDIA AI API — `meta/llama-3.1-8b-instruct` (document analysis)
- Deployed on Hugging Face Spaces (Docker)

## Setup Instructions
1. Clone the repository
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your keys
4. `uvicorn src.main:app --host 0.0.0.0 --port 8000`

## Environment Variables
| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA AI API key from [build.nvidia.com](https://build.nvidia.com) |
| `API_SECRET_KEY` | Secret key for `x-api-key` header authentication |

## API Usage
```
POST /api/document-analyze
Header: x-api-key: your_key
Body: { fileName, fileType, fileBase64 }
```

## Response Format
```json
{
  "status": "success",
  "fileName": "sample.pdf",
  "document_type": "invoice",
  "summary": "...",
  "key_points": ["...", "..."],
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
    "currency": "INR",
    "tax": "",
    "due_date": ""
  },
  "payment_status": "Completed | Pending | Unknown",
  "sentiment": "Positive | Neutral | Negative",
  "confidence": 0.95
}
```

## Supported File Types
| Type | Value |
|---|---|
| PDF | `pdf` |
| Word Document | `docx` |
| Image (JPG/PNG) | `image` |

## Approach
- **PDF**: pdfplumber extracts structured text per page
- **DOCX**: python-docx reads all paragraphs preserving order
- **Image**: pytesseract OCR converts image pixels to text
- **AI Analysis**: Extracted text is sent to NVIDIA AI (LLaMA 3.1-8b-instruct)
  which returns a fully structured JSON with summary, entities, financial details,
  contact info, payment status, sentiment, and confidence score
- **Fallback**: If AI is unavailable, returns extracted text snippet with neutral defaults
