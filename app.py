# Gradio demo: type a message (and optional chat history) and see what
# each model predicts, side by side.
#
# History: one line per message, oldest first, starting with self: / ally: / enemy:
#   ally: why did u dive
#   enemy: ez
# Lines with no prefix count as "ally:".
#
# Run: python app.py          (add --share on Colab to get a public link)
# Models that haven't been trained yet are just skipped.

import sys

import gradio as gr
import joblib
import pandas as pd

sys.path.insert(0, "src")
from common import MODELS_DIR, format_context, has_profanity

# ---- load whatever models exist ----
tfidf = None
if (MODELS_DIR / "tfidf_lr.joblib").exists():
    tfidf = joblib.load(MODELS_DIR / "tfidf_lr.joblib")

toxicbert = None
try:
    from toxicbert_baseline import load_toxicbert, toxicbert_scores
    toxicbert = load_toxicbert()
except Exception as e:
    print("toxic-bert not loaded:", e)

from train_distilbert import load_model, predict_scores
distil_msg = None
distil_ctx = None
if (MODELS_DIR / "distilbert_msg").exists():
    distil_msg = load_model(MODELS_DIR / "distilbert_msg")
if (MODELS_DIR / "distilbert_ctx").exists():
    distil_ctx = load_model(MODELS_DIR / "distilbert_ctx")


def parse_history(history):
    lines = []
    for line in (history or "").splitlines():
        line = line.strip()
        if not line:
            continue
        role, _, text = line.partition(":")
        if role.strip().lower() in ["self", "ally", "enemy"]:
            lines.append((role.strip().lower(), text.strip()))
        else:
            lines.append(("ally", line))
    return format_context(lines[-3:])  # same 3 lines as in training


def classify(message, history):
    message = message.strip()
    if not message:
        return pd.DataFrame(), ""
    context = parse_history(history)

    scores = {}
    if tfidf is not None:
        scores["TF-IDF + LR"] = tfidf.predict_proba([message])[0, 1]
    if toxicbert is not None:
        tb = toxicbert_scores([message], *toxicbert).iloc[0]
        scores["toxic-bert (Rule A)"] = tb["toxic"]
        scores["toxic-bert (Rule B)"] = max(tb["insult"], tb["threat"])
    if distil_msg is not None:
        scores["DistilBERT (message)"] = predict_scores(*distil_msg, [message], [""], False)[0]
    if distil_ctx is not None:
        scores["DistilBERT (context)"] = predict_scores(*distil_ctx, [message], [context], True)[0]

    rows = []
    for name, score in scores.items():
        rows.append({"model": name,
                     "prediction": "HOSTILE" if score >= 0.5 else "not hostile",
                     "score": round(float(score), 3)})
    info = (f"Profanity in message: {'yes' if has_profanity(message) else 'no'}  \n"
            f"Context given to the context model: {context or '(none)'}")
    return pd.DataFrame(rows), info


examples = [
    ["stupid valve", ""],
    ["report mid plz noooob", "ally: mid 0/7\nally: stop feeding"],
    ["nice ult genius", "ally: i missed the ult\nself: again"],
    ["fuck this tree", ""],
    ["noob :D", "ally: lol i died to creeps\nself: haha"],
    ["gg wp", "enemy: gg"],
]

with gr.Blocks(title="Beyond Profanity") as demo:
    gr.Markdown("# Beyond Profanity: Detecting Hostile Intent in Gaming Chat\n"
                "Hostile = aimed at another player. Swearing at the game, at yourself, "
                "or friendly banter should be **not hostile**.")
    with gr.Row():
        with gr.Column():
            message = gr.Textbox(label="Message")
            history = gr.Textbox(label="Chat history (optional, oldest first)", lines=4,
                                 placeholder="ally: mid 0/7\nenemy: ez")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            table = gr.Dataframe(label="Predictions (threshold 0.5)")
            info = gr.Markdown()
    gr.Examples(examples, inputs=[message, history])
    button.click(classify, [message, history], [table, info])
    message.submit(classify, [message, history], [table, info])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
