#!/usr/bin/env bash
# examples.sh — runnable curl examples for all API endpoints
# Run the server first: uvicorn app.main:app --reload
# If API_KEY is set in .env, add: -H "X-Api-Key: <your-key>" to every request

BASE="http://localhost:8000"

echo "=== Upload plain text ==="
curl -s -X POST "$BASE/text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Q1 2024 Strategy Memo",
    "author": "Jane Smith",
    "doc_type": "memo",
    "source": "internal",
    "extra_metadata": {"year": "2024", "quarter": "Q1"},
    "text": "Our Q1 2024 focus is expanding into the Nordic market. We have identified Stockholm and Helsinki as priority cities. The Nordic region shows strong demand for our enterprise product, with an estimated TAM of 200M EUR. Key actions: hire two regional sales leads by February, localise the product for Swedish and Finnish, and establish partnerships with three local system integrators."
  }' | python3 -m json.tool

echo ""
echo "=== Upload a second document (report) ==="
curl -s -X POST "$BASE/text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Nordic Market Research Report 2024",
    "author": "Market Research Team",
    "doc_type": "report",
    "source": "internal",
    "extra_metadata": {"year": "2024", "region": "Nordic"},
    "text": "The Nordic countries — Sweden, Norway, Denmark, and Finland — represent a combined GDP of over 1.5 trillion USD. Enterprise software adoption is among the highest in the world, driven by early digital transformation programs. Key competitors in the region are SAP, Salesforce, and local vendor Visma. Our product differentiates on API-first design and transparent pricing. Customer acquisition cost in the region is estimated at 8,000 EUR per enterprise seat. The recommended go-to-market motion is a partner-led approach via established system integrators."
  }' | python3 -m json.tool

echo ""
echo "=== Upload a PDF file (replace report.pdf with a real file) ==="
# curl -s -X POST "$BASE/document" \
#   -F "file=@report.pdf" \
#   -F "title=Annual Report 2023" \
#   -F "author=Finance Team" \
#   -F "doc_type=report" \
#   -F "source=internal" \
#   -F 'extra_metadata={"year":"2023"}' | python3 -m json.tool

echo "(PDF upload example is commented out — replace report.pdf with a real file to run it)"

echo ""
echo "=== List all documents ==="
curl -s "$BASE/documents" | python3 -m json.tool

echo ""
echo "=== Hybrid search (no filter) ==="
curl -s "$BASE/search?q=Nordic+expansion" | python3 -m json.tool

echo ""
echo "=== Hybrid search filtered to memos only ==="
curl -s "$BASE/search?q=Nordic+expansion&doc_type=memo" | python3 -m json.tool

echo ""
echo "=== Hybrid search filtered by author ==="
curl -s "$BASE/search?q=go-to-market&author=Market+Research+Team" | python3 -m json.tool

echo ""
echo "=== Chat: ask a question (non-streaming) ==="
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is our Q1 strategy for the Nordic market?",
    "doc_type": "memo",
    "stream": false
  }' | python3 -m json.tool

echo ""
echo "=== Chat: ask across all documents (no filter) ==="
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who are the key competitors in the Nordic region?",
    "stream": false
  }' | python3 -m json.tool

echo ""
echo "=== Chat: streaming response (SSE) ==="
echo "(Each line is a Server-Sent Event; final event contains sources)"
curl -s -N -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarise the Nordic go-to-market approach.",
    "stream": true
  }'

echo ""
echo ""
echo "=== Delete a document (replace DOCUMENT_ID with a real UUID from /documents) ==="
# DOCUMENT_ID="00000000-0000-0000-0000-000000000000"
# curl -s -X DELETE "$BASE/documents/$DOCUMENT_ID" -w "HTTP status: %{http_code}\n"
echo "(Delete example is commented out — replace DOCUMENT_ID with a real UUID to run it)"
