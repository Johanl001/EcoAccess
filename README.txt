---
title: EcoAccess_AI
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_file: api.py
app_port: 7860
pinned: false
---

# EcoAccess

AI tool that ingests sustainability content (PDF/URL/Text) and outputs a summary, easy-read bullets, multilingual translation, optional audio, and an infographic card.

## Local setup
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py

## Use
1) Upload a PDF or paste a URL or text
2) Pick Output Language
3) Toggle Audio/Infographic
4) Click Process

## Notes
Educational tool for accessibility. Not official policy guidance.