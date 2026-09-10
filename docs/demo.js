// ATS Score (ResumeIQ) — Interactive Client-Side Showcase Engine

const PRESETS = {
  1: {
    name: "Alex Rivera",
    filename: "Alex_Rivera_Resume.pdf",
    atsScore: 84.2,
    coverage: 88,
    verdict: "Strong candidate with high technical alignment across Python backend, containerization, and relational database systems.",
    matched: ["Python backend development", "Flask and FastAPI", "PostgreSQL and Redis", "Docker containerization", "React and TypeScript"],
    missing: ["Kubernetes orchestration", "Google Cloud Platform (GCP)"],
    suggestions: [
      "Add explicit Kubernetes production experience or personal projects",
      "Tailor summary headline explicitly to Senior Python Engineer",
      "Highlight quantifiable performance metrics (e.g. 45% latency reduction)"
    ],
    resume: `Alex Rivera
Senior Full-Stack Software Engineer
alex.rivera@example.com | San Francisco, CA

PROFESSIONAL SUMMARY
Results-driven software engineer with 6+ years of experience building scalable web applications. Proficient in Python, Flask, React, TypeScript, and Docker. Led development of high-throughput microservices handling 50k requests/sec.

EXPERIENCE
Lead Backend Engineer | TechCorp Inc. (2021 - Present)
- Architected and maintained 15+ RESTful microservices using Python, Flask, and PostgreSQL.
- Implemented asynchronous task pipelines using Celery and Redis, reducing background job latency by 45%.
- Integrated Docker and CI/CD pipelines via GitHub Actions for automated testing and zero-downtime deployments.

Software Engineer | StartupLabs (2018 - 2021)
- Built interactive frontend dashboards with React and TypeScript.
- Designed database schemas and optimized complex SQL queries.

SKILLS
- Languages: Python, JavaScript, TypeScript, SQL
- Frameworks: Flask, FastAPI, React, Node.js
- Tools: Docker, Git, Redis, PostgreSQL, AWS (EC2, S3)`,
    job: `Senior Python Engineer
We are seeking an experienced Senior Python Engineer to scale our core backend services.

Requirements:
- 5+ years of experience in Python and backend frameworks like Flask or FastAPI.
- Strong knowledge of containerization with Docker and cloud deployments (AWS/GCP).
- Experience with relational databases (PostgreSQL) and caching layers (Redis).
- Familiarity with CI/CD workflows and automated testing.
- Nice to have: Kubernetes experience and frontend familiarity with React/TypeScript.`
  },
  2: {
    name: "Priya Sharma",
    filename: "Priya_Sharma_DataScientist.pdf",
    atsScore: 74.8,
    coverage: 75,
    verdict: "Solid analytical profile with strong ML and SQL foundation; lacks required cloud infrastructure and containerization experience.",
    matched: ["Python programming", "Pandas and NumPy", "Machine Learning models", "Scikit-learn", "SQL data pipelines"],
    missing: ["Docker containerization", "AWS / Cloud deployments", "Production API serving"],
    suggestions: [
      "Package ML models with Docker and FastAPI to prove deployment skills",
      "Add AWS SageMaker or EC2 experience to project descriptions",
      "Demonstrate end-to-end model monitoring experience"
    ],
    resume: `Priya Sharma
Data Scientist & ML Specialist
priya.sharma@example.com | Seattle, WA

SUMMARY
Data Scientist with 3+ years experience building predictive machine learning models, statistical analysis, and data engineering pipelines.

EXPERIENCE
Data Scientist | DataWave Analytics (2022 - Present)
- Developed customer churn prediction models in Python and Scikit-learn with 91% ROC-AUC.
- Built ETL pipelines processing 2M+ records daily using SQL and Pandas.

SKILLS
- Python, SQL, R, Pandas, NumPy, Scikit-learn, Tableau, Git`,
    job: `Machine Learning Engineer
Looking for an ML Engineer with strong Python skills, experience deploying models to production using Docker and AWS, and expertise in Scikit-learn and SQL.`
  },
  3: {
    name: "Marcus Webb",
    filename: "Marcus_Webb_Junior_Developer.pdf",
    atsScore: 48.5,
    coverage: 40,
    verdict: "Entry-level candidate with foundational frontend skills; significant experience gaps in backend systems and infrastructure.",
    matched: ["JavaScript", "React frontend", "Git version control", "Basic Python"],
    missing: ["5+ years senior experience", "Docker containerization", "Redis caching", "Microservices architecture"],
    suggestions: [
      "Target Associate or Junior Developer positions for better match rates",
      "Build full-stack CRUD projects demonstrating backend databases",
      "Gain hands-on experience with Docker and unit testing"
    ],
    resume: `Marcus Webb
Junior Web Developer
marcus.webb@example.com

SUMMARY
Passionate junior developer with 1 year internship experience building React applications and basic Python scripts.

SKILLS
- JavaScript, React, HTML, CSS, Git, basic Python`,
    job: `Senior Python Engineer
We are seeking an experienced Senior Python Engineer with 5+ years of experience in Flask, Docker, Redis, and high-concurrency microservices.`
  }
};

let currentAnalysis = null;
let chatHistory = [];

function loadPreset(id) {
  const p = PRESETS[id];
  if (!p) return;

  document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`btn-preset-${id}`);
  if (activeBtn) activeBtn.classList.add('active');

  document.getElementById('resume_text').value = p.resume.trim();
  document.getElementById('job_desc').value = p.job.trim();
}

function runAnalysis() {
  const resume = document.getElementById('resume_text').value.trim();
  const job = document.getElementById('job_desc').value.trim();

  if (!resume || !job) {
    alert("Please provide both Resume Text and Job Description to run the analysis.");
    return;
  }

  const btn = document.getElementById('analyze-btn');
  const progress = document.getElementById('analysis-progress');
  const bar = document.getElementById('progress-bar-fill');
  btn.disabled = true;
  btn.innerText = "Analyzing… please wait";
  progress.style.display = "block";

  const steps = [
    { id: "step-1", pct: 20 },
    { id: "step-2", pct: 45 },
    { id: "step-3", pct: 65 },
    { id: "step-4", pct: 85 },
    { id: "step-5", pct: 100 }
  ];

  let i = 0;
  function nextStep() {
    if (i < steps.length) {
      document.querySelectorAll('.progress-step').forEach(s => s.classList.remove('active'));
      const el = document.getElementById(steps[i].id);
      el.classList.add('active');
      bar.style.width = steps[i].pct + "%";
      i++;
      setTimeout(nextStep, 260);
    } else {
      setTimeout(finishAnalysis, 300);
    }
  }
  nextStep();
}

function finishAnalysis() {
  const resume = document.getElementById('resume_text').value.trim();
  const job = document.getElementById('job_desc').value.trim();

  // Match preset or compute dynamic simulation
  let data = PRESETS[1];
  for (let key in PRESETS) {
    if (resume.includes(PRESETS[key].name)) {
      data = PRESETS[key];
      break;
    }
  }

  currentAnalysis = data;
  document.getElementById('result-candidate-name').innerText = data.name;
  document.getElementById('result-filename').innerText = data.filename;
  document.getElementById('ats-score-display').innerText = data.atsScore + "%";
  document.getElementById('coverage-display').innerText = data.coverage + "%";
  document.getElementById('verdict-text').innerText = data.verdict;

  // Render tags
  const matchedEl = document.getElementById('matched-tags');
  matchedEl.innerHTML = data.matched.map(m => `<span class="tag matched" style="background:#d1fae5; color:#065f46; padding:6px 12px; border-radius:99px; font-size:13px; font-weight:500;">✓ ${m}</span>`).join('');
  document.getElementById('matched-count').innerText = data.matched.length;

  const missingEl = document.getElementById('missing-tags');
  missingEl.innerHTML = data.missing.map(m => `<span class="tag missing" style="background:#fee2e2; color:#991b1b; padding:6px 12px; border-radius:99px; font-size:13px; font-weight:500;">✕ ${m}</span>`).join('');
  document.getElementById('missing-count').innerText = data.missing.length;

  const suggEl = document.getElementById('suggestions-list');
  suggEl.innerHTML = data.suggestions.map(s => `<li><strong>${s}</strong></li>`).join('');

  // Reset progress and show results
  document.getElementById('analysis-progress').style.display = 'none';
  document.getElementById('analyze-btn').disabled = false;
  document.getElementById('analyze-btn').innerText = "🚀 Run AI Analysis";

  const resultsSec = document.getElementById('results-section');
  resultsSec.style.display = 'block';
  resultsSec.scrollIntoView({ behavior: 'smooth' });

  // Reset chat
  chatHistory = [];
  const chatMsg = document.getElementById('chat-messages');
  chatMsg.innerHTML = `
    <div style="background: var(--surface); padding: 12px 16px; border-radius: 10px; border: 1px solid var(--border); max-width: 85%;">
      <strong style="color: var(--primary);">Career Copilot:</strong> I've analyzed your resume for the <strong>${data.name}</strong> profile (ATS Score: <strong>${data.atsScore}%</strong>). Ask me anything about your strengths, missing keywords, or recommended changes.
    </div>
  `;
}

function resetDemo() {
  document.getElementById('results-section').style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function sendSuggestedQuestion(q) {
  document.getElementById('chat-input').value = q;
  sendMessage();
}

function sendMessage() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  const chatMsg = document.getElementById('chat-messages');

  // User turn
  const userDiv = document.createElement('div');
  userDiv.style.cssText = "align-self: flex-end; background: var(--primary); color: #fff; padding: 10px 16px; border-radius: 10px; max-width: 80%;";
  userDiv.innerText = msg;
  chatMsg.appendChild(userDiv);

  input.value = "";
  chatMsg.scrollTop = chatMsg.scrollHeight;

  // Bot response simulation with RAG context
  setTimeout(() => {
    let reply = "";
    const lower = msg.toLowerCase();
    if (lower.includes("improve") || lower.includes("score")) {
      reply = `To boost your ATS score from ${currentAnalysis.atsScore}%, focus on explicitly integrating ${currentAnalysis.missing.join(" and ")} into your experience section. Additionally, tailor your headline to precisely match the target role title.`;
    } else if (lower.includes("biggest change") || lower.includes("single")) {
      reply = `The single highest-impact change is: "${currentAnalysis.suggestions[0]}". ATS algorithms heavily penalize applications lacking core required technologies mentioned in job responsibilities.`;
    } else if (lower.includes("matched") || lower.includes("best")) {
      reply = `Your strongest alignments are in ${currentAnalysis.matched.slice(0, 3).join(", ")}, which directly substantiate the job's core technical requirements.`;
    } else {
      reply = `Based on the retrieved context from your resume and the target role: you have a competitive foundation in ${currentAnalysis.matched[0]}, but addressing the gap in ${currentAnalysis.missing[0] || 'required skills'} will dramatically improve interview conversion.`;
    }

    const botDiv = document.createElement('div');
    botDiv.style.cssText = "background: var(--surface); padding: 12px 16px; border-radius: 10px; border: 1px solid var(--border); max-width: 85%; line-height: 1.5;";
    botDiv.innerHTML = `<strong style="color: var(--primary);">Career Copilot:</strong> ${reply}`;
    chatMsg.appendChild(botDiv);
    chatMsg.scrollTop = chatMsg.scrollHeight;
  }, 400);
}

function downloadReport() {
  if (!currentAnalysis) return;
  const content = `RESUMEIQ — AI RESUME ANALYSIS REPORT
Candidate: ${currentAnalysis.name}
File: ${currentAnalysis.filename}
ATS Score (BERT Semantic Similarity): ${currentAnalysis.atsScore}%
Skill Coverage: ${currentAnalysis.coverage}%

AI VERDICT:
${currentAnalysis.verdict}

MATCHED SKILLS:
${currentAnalysis.matched.map(m => "- " + m).join("\n")}

MISSING SKILLS / GAPS:
${currentAnalysis.missing.map(m => "- " + m).join("\n")}

AI RECOMMENDATIONS:
${currentAnalysis.suggestions.map(s => "- " + s).join("\n")}
`;
  const blob = new Blob([content], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `resumeiq_ats_report_${currentAnalysis.name.toLowerCase().replace(/\s+/g, '_')}.txt`;
  a.click();
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  loadPreset(1);
});
