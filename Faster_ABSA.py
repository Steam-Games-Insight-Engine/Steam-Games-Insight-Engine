
# ✅ EXPANDED ABSA METHOD: Keyword-Based Aspect Detection + Transformer Sentiment + Aggregation

import pandas as pd
import re
import numpy as np
from transformers import pipeline
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# data processing...
df = pd.read_csv("data/Black Myth WuKong Reviews.csv")
df = df.dropna(subset=['review'])

# aspects + keywords
aspect_keywords = {
    "graphics": ["graphics", "visuals", "art", "animation", "design", "style", "textures", "shader", "model"],
    "combat": ["combat", "fighting", "battle", "enemy", "attack", "mechanics", "weapons", "boss", "action"],
    "performance": ["lag", "fps", "frame", "performance", "crash", "glitch", "bug", "optimize", "loading", "stutter"],
    "story": ["story", "plot", "narrative", "dialogue", "characters", "ending", "cutscene", "lore"],
    "controls": ["controls", "movement", "input", "button", "responsive", "smooth"],
    "audio": ["sound", "music", "audio", "voice", "effects", "soundtrack"],
    "multiplayer": ["multiplayer", "online", "pvp", "coop", "team", "matchmaking", "friends"],
    "level_design": ["level", "map", "environment", "structure", "flow", "design", "pacing"]
}

# sentiment pipeline
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def extract_aspects(text):
    found = set()
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                found.add(aspect)
    return list(found)

# progress of reviews being analyzed
aspect_sentiments = defaultdict(list)
aspect_confidence = defaultdict(list)
aspect_counts = Counter()

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Reviews"):
    review = row['review']
    aspects = extract_aspects(review)
    for aspect in aspects:
        aspect_counts[aspect] += 1
    if not aspects:
        continue

    try:
        sentiment_result = sentiment_model(review[:512])[0]
        label = sentiment_result['label']
        score = sentiment_result['score']
        if score < 0.6:
            continue
    except Exception as e:
        continue

    for aspect in aspects:
        aspect_sentiments[aspect].append(label)
        aspect_confidence[aspect].append(score)

# aggregate results
summary = {}
for aspect in aspect_sentiments:
    sentiments = aspect_sentiments[aspect]
    confidences = aspect_confidence[aspect]
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")
    total = pos + neg
    if total == 0:
        continue
    avg_conf = round(np.mean(confidences), 3)
    summary[aspect] = {
        "mentions": aspect_counts[aspect],
        "positive": pos,
        "negative": neg,
        "total_analyzed": total,
        "percent_positive": round(pos / total * 100, 2),
        "avg_confidence": avg_conf
    }

# visualization
summary_df = pd.DataFrame(summary).T.sort_values("percent_positive", ascending=False)
summary_df[["positive", "negative"]].plot(kind="bar", figsize=(12, 6), title="Aspect Sentiment Breakdown")
plt.ylabel("Review Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# table of the final sentiments
print("\Aspect Sentiments:")
print(summary_df[["mentions", "positive", "negative", "total_analyzed", "percent_positive", "avg_confidence"]])
