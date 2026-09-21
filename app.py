"""
ATS Score Application Entry Point.
Production runner for Render Cloud (Telegram Webhook & HTTP Healthcheck Server)
and optional Gradio Web Interface for Hugging Face Spaces.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Force CPU mode globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"


def main():
    # Detect environment:
    # Render sets RENDER=true, RENDER_EXTERNAL_URL, PORT.
    # Default to running the unified Webhook & Healthcheck bot on Render/production.
    is_render = bool(os.getenv("RENDER") or os.getenv("RENDER_EXTERNAL_URL") or os.getenv("WEBHOOK_URL"))
    run_mode = os.getenv("RUN_MODE", "").lower()
    is_hf_spaces = bool(os.getenv("SPACE_ID"))

    if is_render or run_mode == "bot" or not is_hf_spaces:
        # Production Telegram Webhook + HTTP Healthcheck Server on 0.0.0.0:$PORT
        import bot
        bot.main()
    else:
        # Fallback for Hugging Face Spaces Gradio Web UI
        _launch_gradio()


def _launch_gradio():
    try:
        import gradio as gr
    except ImportError:
        print("[!] Gradio is not installed in this environment. Running Telegram Bot instead...", flush=True)
        import bot
        bot.main()
        return

    from services.pdf_service import extract_resume_text
    from services.similarity_service import calculate_similarity
    from services.gemini_service import generate_report, compute_blended_ats_score
    from config import Config

    def analyze_resumes(resume_files, resume_text_input, jd_text):
        if not jd_text or len(jd_text.strip()) < 20:
            return "### ⚠️ Please provide a detailed Job Description (minimum 20 characters)."

        resumes_to_process = []
        if resume_files:
            for rf in resume_files:
                try:
                    with open(rf.name, "rb") as f:
                        txt = extract_resume_text(f)
                        resumes_to_process.append({"filename": os.path.basename(rf.name), "text": txt})
                except Exception as e:
                    return f"### ❌ Failed to parse PDF {os.path.basename(rf.name)}: {e}"
        elif resume_text_input and len(resume_text_input.strip()) > 20:
            resumes_to_process.append({"filename": "Pasted_Resume.txt", "text": resume_text_input.strip()})
        else:
            return "### ⚠️ Please upload at least one Resume PDF or paste your resume text."

        results = []
        for r in resumes_to_process:
            rep = generate_report(r["text"], jd_text)
            sim = calculate_similarity(r["text"], jd_text)
            score, rating = compute_blended_ats_score(rep, sim)
            results.append({
                "filename": r["filename"],
                "ats_score": score,
                "rating": rating,
                "report": rep,
            })

        results.sort(key=lambda x: x["ats_score"], reverse=True)

        if len(results) == 1:
            top = results[0]
            rep = top["report"]
            matched = "\n".join(f"• {s}" for s in rep.get("matched_skills", [])) or "None detected"
            missing = "\n".join(f"• {s}" for s in rep.get("missing_skills", [])) or "None detected"
            sugg = "\n".join(f"• {s}" for s in rep.get("suggestions", [])) or "None"
            blocks = min(10, max(0, int(round(top["ats_score"] / 10))))
            gauge = "█" * blocks + "░" * (10 - blocks)

            return f"""## 🎯 ATS Match Score: {top['ats_score']}% — *{top['rating']}*
`{gauge}`

### 📋 Recruiter Verdict:
{rep.get('verdict', 'Analysis complete.')}

---

### ✅ Matched Skills ({len(rep.get('matched_skills', []))}):
{matched}

---

### ❌ Missing Skills / Keywords ({len(rep.get('missing_skills', []))}):
{missing}

---

### 💡 Actionable Recommendations:
{sugg}
"""

        output_lines = [
            f"## 🏆 ATS Candidate Rankings ({len(results)} Resumes Analyzed)",
            "| Rank | Candidate File | ATS Match Score | Rating | Top Matched Skills |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        medals = ["🥇 1", "🥈 2", "🥉 3"]
        for idx, r in enumerate(results, start=1):
            m = medals[idx - 1] if idx <= 3 else f"#{idx}"
            top_skills = ", ".join(r["report"].get("matched_skills", [])[:3]) or "None"
            output_lines.append(f"| **{m}** | `{r['filename']}` | **{r['ats_score']}%** | *{r['rating']}* | {top_skills} |")

        output_lines.append("\n---\n\n### 📄 Detailed Candidate Audit Reports:\n")

        for idx, r in enumerate(results, start=1):
            rep = r["report"]
            matched = "\n".join(f"• {s}" for s in rep.get("matched_skills", [])) or "None detected"
            missing = "\n".join(f"• {s}" for s in rep.get("missing_skills", [])) or "None detected"
            sugg = "\n".join(f"• {s}" for s in rep.get("suggestions", [])) or "None"
            blocks = min(10, max(0, int(round(r["ats_score"] / 10))))
            gauge = "█" * blocks + "░" * (10 - blocks)

            output_lines.append(f"""#### #{idx}: `{r['filename']}` — **{r['ats_score']}%** (*{r['rating']}*)
`{gauge}`
*Verdict: {rep.get('verdict', '')}*

**Matched Skills:**
{matched}

**Missing Skills / Keywords:**
{missing}

**Recommendations:**
{sugg}

---
""")

        return "\n".join(output_lines)

    with gr.Blocks(title="ATS Score & Telegram Career Copilot", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("# 🤖 AI Resume ATS Scorer & Telegram Career Copilot")
        gr.Markdown(
            """
            > 🟢 **Telegram Bot is ACTIVE 24/7!** You can chat directly on Telegram with **[@MyResumeIQBot](https://t.me/MyResumeIQBot)**.
            
            Upload **1 or more resume PDFs** (or paste text) along with the target Job Description to compare candidate rankings and get instant skill gap audits!
            """
        )

        with gr.Row():
            with gr.Column():
                resume_files = gr.File(label="📄 Upload Resume(s) (PDF) — Supports Multiple Files", file_count="multiple", file_types=[".pdf"])
                resume_text = gr.Textbox(label="Or Paste Resume Text (Single Candidate)", lines=5, placeholder="Paste your resume text here...")
                jd_text = gr.Textbox(label="📝 Target Job Description", lines=6, placeholder="Paste the job description requirements here...")
                submit_btn = gr.Button("🚀 Analyze & Rank Resumes", variant="primary")

            with gr.Column():
                output_markdown = gr.Markdown(value="*Upload resume(s) and paste a JD to see the rankings and audit report.*")

        submit_btn.click(
            fn=analyze_resumes,
            inputs=[resume_files, resume_text, jd_text],
            outputs=[output_markdown]
        )

    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
