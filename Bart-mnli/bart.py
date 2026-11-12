
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

labels = [
    "the speaker themselves",
    "another player",
    "the entire team",
    "the video game or its environment",
    "a technical problem",
    "nobody in particular"
]

messages = [
    "I can't believe how bad I played today.",
    "This game is terrible.",
    "That player is so annoying.",
    "Our team is a mess.",
    "I'm not mad at anyone.",
    "I'm such an idiot.",
    "stupid tree",
    "my shit is lagging",
    "Great game guys",
    "Player 2 is absolutely dog shit",
    "Can't believe I missed that shot"
]

for message in messages:
    result = classifier(
        message,
        candidate_labels=labels,
        hypothesis_template="The criticism is directed at {}.",
        multi_label=False
    )

    print(f"\nMessage: {message}")
    for label, score in zip(
        result["labels"][:3],
        result["scores"][:3]
    ):
        print(f"  {score:.2%} | {label}")
