"""
Gradio Web Interface + 24/7 Telegram Bot Runner for Hugging Face Spaces.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import threading
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ZeroGPU compatibility for Hugging Face Spaces
try:
    import spaces

    @spaces.GPU(duration=1)
    def dummy_gpu():
        """Satisfies ZeroGPU startup check on Hugging Face Spaces."""
        return None

    print("[*] ZeroGPU detected: registered @spaces.GPU hook successfully.", flush=True)
except ImportError:
    pass


import bot
from services.pdf_service import extract_resume_text
from services.similarity_service import calculate_similarity
from services.gemini_service import generate_report, compute_blended_ats_score

# Start the Telegram Bot in a background thread
print("[*] Starting Telegram Bot in background worker thread...", flush=True)
bot_thread = threading.Thread(target=bot.main, kwargs={"in_thread": True}, daemon=True, name="telegram-bot-runner")
bot_thread.start()


def analyze_resume(resume_file, resume_text_input, jd_text):
    if not jd_text or len(jd_text.strip()) < 20:
        return "### ⚠️ Please provide a detailed Job Description (minimum 20 characters)."

    extracted_text = ""
    if resume_file is not None:
        try:
            with open(resume_file.name, "rb") as f:
                extracted_text = extract_resume_text(f)
        except Exception as e:
            return f"### ❌ Failed to parse PDF: {e}"
    elif resume_text_input and len(resume_text_input.strip()) > 20:
        extracted_text = resume_text_input.strip()
    else:
        return "### ⚠️ Please upload a Resume PDF or paste your resume text."

    # Step 1: Gemini deep audit (verifies skills, missing keywords, and role alignment)
    report = generate_report(extracted_text, jd_text)

    # Step 2: BERT semantic similarity on CPU
    similarity = calculate_similarity(extracted_text, jd_text)

    # Step 3: Compute honest, skill-grounded ATS score
    ats_score, rating = compute_blended_ats_score(report, similarity)

    matched = "\n".join(f"• {s}" for s in report.get("matched_skills", [])) or "None detected"
    missing = "\n".join(f"• {s}" for s in report.get("missing_skills", [])) or "None detected"
    sugg = "\n".join(f"• {s}" for s in report.get("suggestions", [])) or "None"

    # Gauge
    blocks = min(10, max(0, int(round(ats_score / 10))))
    gauge = "█" * blocks + "░" * (10 - blocks)

    output = f"""## 🎯 ATS Match Score: {ats_score}% — *{rating}*
`{gauge}`

### 📋 Recruiter Verdict:
{report.get('verdict', 'Analysis complete.')}

---

### ✅ Matched Skills ({len(report.get('matched_skills', []))}):
{matched}

---

### ❌ Missing Skills / Keywords ({len(report.get('missing_skills', []))}):
{missing}

---

### 💡 Actionable Recommendations:
{sugg}
"""
    return output


# Build clean modern Gradio UI
with gr.Blocks(title="ATS Score & Telegram Career Copilot", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🤖 AI Resume ATS Scorer & Telegram Career Copilot")
    gr.Markdown(
        """
        > 🟢 **Telegram Bot is ACTIVE 24/7!** You can chat directly on Telegram with **[@MyResumeIQBot](https://t.me/MyResumeIQBot)**.
        
        Evaluates your resume against target job description requirements using **verified skill matching**, **role alignment auditing**, and **Google Gemini analysis**.
        """
    )

    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"])
            resume_text = gr.Textbox(label="Or Paste Resume Text", lines=5, placeholder="Paste your resume text here...")
            jd_text = gr.Textbox(label="📝 Target Job Description", lines=6, placeholder="Paste the job description requirements here...")
            submit_btn = gr.Button("🚀 Analyze Resume", variant="primary")

        with gr.Column():
            output_markdown = gr.Markdown(value="*Submit your resume and JD to see the audit report.*")

    submit_btn.click(
        fn=analyze_resume,
        inputs=[resume_file, resume_text, jd_text],
        outputs=[output_markdown]
    )

    gr.Markdown(
        """
        ---
        💬 **Want to chat with the AI Career Copilot?** Open Telegram and message **[@MyResumeIQBot](https://t.me/MyResumeIQBot)** to ask questions like *"How can I improve my score?"* or *"Why is Docker missing?"*
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
