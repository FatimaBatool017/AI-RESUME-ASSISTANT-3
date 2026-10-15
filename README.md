# AI-RESUME-ASSISTANT-3
# 📄 ATS Resume Checker

Upload a resume (PDF or DOCX) and get an **ATS score (0-100)**, a category breakdown, missing keywords, and prioritised improvements with example rewrites.

Built with **Streamlit** (UI) and **Google Gemini Flash** (AI).

## Features

- PDF and DOCX support (including resumes laid out in tables)
- Optional job description for a more accurate keyword score
- Score breakdown: keywords, formatting, content quality, structure, readability
- Strengths, weaknesses, missing keywords, prioritised fixes, before/after rewrites
- Instant rule-based checks (email, phone, sections, length, quantified results)
- Automatic fallback between Gemini models if one is unavailable

## Run locally

1. Get a free API key: https://aistudio.google.com/apikey
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Add your key (pick one):

**Option A: secrets file (recommended)**

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

**Option B: environment variable**

```bash
export GEMINI_API_KEY="your-key-here"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

**Option C:** just paste the key into the app's sidebar.

4. Start the app:

```bash
streamlit run app.py
```

## Optional: choose the model

Add this to secrets (or set as an environment variable) to try a specific model first:

```toml
GEMINI_MODEL = "gemini-2.5-flash"
```

## Push to GitHub

```bash
git init
git add app.py README.md requirements.txt .gitignore
git commit -m "Initial commit: ATS resume checker"
git branch -M main
git remote add origin https://github.com/<your-username>/ats-resume-checker.git
git push -u origin main
```

Create the empty repository on github.com first (no README or .gitignore added there).

**Important:** never commit `.streamlit/secrets.toml`. Use this `.gitignore`:

```
venv/
__pycache__/
.streamlit/secrets.toml
.env
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose your repository, branch `main`, and main file `app.py`.
3. Open **Advanced settings** and paste into **Secrets**:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

4. Click **Deploy**. You get a public URL in a minute or two.

Later pushes to `main` redeploy automatically.

## Notes

- ATS scores here are AI estimates, not the output of a real employer's system.
- Scanned or image-only resumes can't be read; export a text-based PDF or DOCX.
- Resumes are sent to the Gemini API for analysis and are not stored by this app.
