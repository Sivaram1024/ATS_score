# 🧠 GenAI Resume Analyzer

An AI-powered Resume Analyzer that evaluates resumes against job descriptions using semantic similarity and Google Gemini AI. The application extracts text from PDF resumes, calculates an ATS similarity score using Sentence Transformers, and generates intelligent feedback to help candidates improve their resumes.

---

## 🚀 Features

- 📄 Upload Resume (PDF)
- 💼 Enter Job Description
- 📊 ATS Similarity Score
- 🤖 AI-powered Resume Evaluation
- ✅ Skills Matched
- ❌ Missing Skills
- 🔄 Partial Matches
- 💡 Resume Improvement Suggestions
- ⚡ Interactive Streamlit Interface

---

## 🛠 Tech Stack

### Programming Language
- Python

### Framework
- Streamlit

### AI & NLP
- Google Gemini API
- Sentence Transformers
- all-mpnet-base-v2
- BERT Embeddings
- Cosine Similarity

### Libraries
- pdfminer.six
- scikit-learn
- python-dotenv
- nest_asyncio

---

## ⚙️ Workflow

1. Upload Resume (PDF)
2. Extract text from PDF
3. Enter Job Description
4. Generate semantic embeddings using Sentence Transformers
5. Calculate cosine similarity between resume and job description
6. Generate ATS Similarity Score
7. Generate AI feedback using Google Gemini
8. Display:
   - Skills Matched
   - Missing Skills
   - Partial Matches
   - Resume Improvement Suggestions

---

## 🏗 Project Architecture

```
               Resume PDF
                    │
                    ▼
         PDF Text Extraction
                    │
                    ▼
      Sentence Transformer Model
                    │
                    ▼
         Semantic Embeddings
                    │
                    ▼
         Cosine Similarity Score
                    │
                    ▼
              ATS Score
                    │
                    ▼
         Google Gemini AI
                    │
                    ▼
         Resume Feedback Report
```

---

## 📂 Project Structure

```
GenAI-Resume-Analyzer/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── screenshots/
    ├── home.png
    ├── upload.png

```

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/CheshtaSharma/GenAI-Resume-Analyzer.git
```

Navigate into the project

```bash
cd GenAI-Resume-Analyzer
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GEMINI_API_KEY=YOUR_API_KEY
```

Run the application

```bash
streamlit run main.py
```

---

## 📸 Screenshots

### Home Page

<img width="1782" height="729" alt="image" src="https://github.com/user-attachments/assets/56ad7eb9-7bd7-4e89-accf-742b1318e4ff" />


### Resume Upload

<img width="1771" height="653" alt="image" src="https://github.com/user-attachments/assets/d5056a3f-519c-4d0f-9d38-7f76f2ae6f17" />


---

## 📚 Learning Outcomes

This project helped me understand:

- Semantic Search
- Sentence Transformers
- BERT Embeddings
- Cosine Similarity
- Prompt Engineering
- Google Gemini API Integration
- PDF Text Extraction
- Streamlit Application Development
- Environment Variable Management

---

## 🔮 Future Enhancements

- Resume Ranking
- ATS Optimization Report
- Cover Letter Generator
- Interview Question Generator
- Multi-language Resume Analysis
- Recruiter Dashboard
- Resume Keyword Optimization

---

## 👩‍💻 Author

**Cheshta Sharma**

B.Tech Computer Science (Artificial Intelligence)

Python | Machine Learning | Generative AI | Data Analytics
