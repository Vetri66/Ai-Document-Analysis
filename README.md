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
REST API that accepts PDF, DOCX, and image files as base64,
extracts text, and uses Google Gemini AI for summarisation,
entity extraction, and sentiment analysis.

## Tech Stack
- Python + FastAPI
- pdfplumber (PDF extraction)
- python-docx (DOCX extraction)
- pytesseract + Pillow (OCR for images)
- Google Gemini API (AI analysis)

## Setup Instructions
1. Clone the repository
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your keys
4. `uvicorn src.main:app --host 0.0.0.0 --port 8000`

## API Usage
```
POST /api/document-analyze
Header: x-api-key: your_key
Body: { fileName, fileType, fileBase64 }
```

## Approach
- PDF: pdfplumber extracts structured text per page
- DOCX: python-docx reads all paragraphs preserving order
- Image: pytesseract OCR converts image pixels to text
- All extracted text is sent to Google Gemini (gemini-2.0-flash)
  which returns summary, entities, and sentiment as JSON
