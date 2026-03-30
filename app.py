"""
app.py
Gradio UI for the IIT KGP Course Planning Assistant.

Run with:
    python app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")
print("OPENAI_API_KEY loaded:", openai_key[:10] + "...")  # partial print for safety

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
import json
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CORE CHAT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def chat(user_message: str, history: list) -> tuple[str, list]:
    if not user_message.strip():
        return "", history

    result = run_pipeline(user_message, verbose=False)
    response = result["final_response"]

    # Append verdict badge if verifier ran
    if result["verdict"] not in ("INCOMPLETE", ""):
        verdict_map = {
            "PASS": "Verified",
            "FAIL": "Unverified claims removed",
            "NEEDS REVISION": "Revised for accuracy",
        }
        badge = verdict_map.get(result["verdict"], "")
        if badge:
            response += f"\n\n_{badge}_"

    history.append((user_message, response))
    return "", history


def show_debug(user_message: str) -> str:
    if not user_message.strip():
        return "Enter a message first."

    result = run_pipeline(user_message, verbose=False)

    lines = []
    lines.append(f"VERDICT: {result['verdict']}")
    lines.append(f"\nISSUES:\n{result.get('issues', 'None')}")
    lines.append(f"\nPROFILE:\n{json.dumps(result['profile'], indent=2)}")
    lines.append(f"\nRETRIEVED CHUNKS ({len(result['context_chunks'])}):")
    for c in result["context_chunks"][:5]:
        lines.append(f"  [{c['score']:.3f}] {c['source']} / {c['section']}")
        lines.append(f"  {c['text'][:150]}...")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="IIT KGP Course Planner") as demo:

    gr.Markdown("""
    # IIT KGP Course Planning Assistant
    ### CSE Minor Planning for ISE Students
    
    Ask about prerequisites, course eligibility, or get a full term plan.
    All answers are grounded in the IIT KGP catalog with citations.
    
    **Example questions:**
    - *"I am an ISE student in 5th sem, CGPA 8.5, no backlogs. I've done IM21003, CS21001. What CSE minor courses can I take?"*
    - *"Can I take CS31001 if I've completed CS31007?"*
    - *"What are the CGPA requirements to register for the CSE minor?"*
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type your question here...",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "I am an ISE student in 5th semester with CGPA 8.5 and no backlogs. I have completed IM21003, IM21001, MA20104, CS21001, CS21003. What CSE minor courses can I take next semester?",
                    "Can I take CS31001 Operating Systems if I have completed CS31007?",
                    "What is the minimum CGPA to register for CSE minor?",
                    "I have CGPA 7.2. Can I do the CSE minor?",
                    "What courses are required for the CSE minor at IIT KGP?",
                    "Is CS29008 DBMS Lab available without Discrete Structures?",
                ],
                inputs=msg_box,
            )

        with gr.Column(scale=2):
            gr.Markdown("### Debug Panel")
            debug_btn = gr.Button("Run Debug on Last Message")
            debug_box = gr.Textbox(
                label="Retrieved chunks + verification details",
                lines=25,
                max_lines=30,
            )

    # ── Event handlers
    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    )
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    )
    debug_btn.click(
        fn=lambda history: show_debug(history[-1][0]) if history else "No messages yet.",
        inputs=[chatbot],
        outputs=[debug_box],
    )

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True)