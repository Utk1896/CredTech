# CredTech: Corporate Credit Risk Platform(Hackathon Submission)

## Overview
CredTech is a real-time corporate credit risk analysis platform designed to evaluate the financial health and default probability of publicly traded S&P 500 companies. The system synthesizes quantitative fundamental financial data with qualitative market sentiment to generate a unified risk metric known as the NexScore.

## Architecture & Tech Stack
- **Database:** PostgreSQL (Neon)
- **Data Engineering:** Python, pandas, yfinance
- **Machine Learning:** LightGBM (Gradient-Boosting Classifier), scikit-learn
- **Generative AI:** Google Gemini 1.5 API
- **Backend API:** FastAPI, Uvicorn, psycopg2
- **Frontend UI:** Next.js, React, Tailwind CSS

## Core Components
1. **Data Ingestion Pipeline:** Extracts and standardizes 10 years of SEC financial filings via the Yahoo Finance API to calculate core leverage ratios.
2. **Asynchronous Worker:** Deploys a LightGBM classifier to evaluate fundamental ratios and calculates a default probability score. Concurrently integrates the Gemini 1.5 API to extract structured market sentiment from breaking financial news.
3. **Web Dashboard:** A Next.js application served by a RESTful FastAPI backend utilizing server-side pagination to visualize the synthesized NexScores.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Node.js (v18+) and npm
- PostgreSQL Database
- Gemini API Key and News API Key

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://user:password@endpoint.neon.tech/neondb
GEMINI_API_KEY=your_gemini_key_here
NEWS_KEY=your_news_api_key_here
```

### 3. Installation
Initialize the Python virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Execution
Run the system components in separate terminal instances:

**Data Ingestion:**
```bash
cd worker
python build_structured_features.py
```

**Background Worker:**
```bash
cd worker
python run.py --mode update
```

**Backend API:**
```bash
cd backend
source ../venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Frontend Dashboard:**
```bash
cd frontend-next
npm install
npm run dev
```

The application will be accessible at `http://localhost:3000`.
