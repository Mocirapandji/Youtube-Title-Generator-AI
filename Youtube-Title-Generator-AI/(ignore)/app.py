import os
import joblib
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────────────────────────
SAVE_PATH       = "/home/s5812886/Desktop/Youtube-Title-Generator-AI/AllCode"
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "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")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyA5m2QMKaUP7yQ0X_cKSKGZn5kbxwDDur8")
MUSIC_CATEGORY  = "10"

# ── Load models once ────────────────────────────────────────────────────────
print("Loading models…")
xgb_model    = joblib.load(f"{SAVE_PATH}/youtube_model.pkl")
embedder     = SentenceTransformer('all-MiniLM-L6-v2')
groq_client  = Groq(api_key=GROQ_API_KEY)
youtube      = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
print("All systems ready.")

# ── Helpers ─────────────────────────────────────────────────────────────────
def brainstorm(idea):
    prompt = (
        f"Generate 10 clickable YouTube titles for: '{idea}'. "
        "Use curiosity gaps and emotional hooks. No numbers or bullets, just titles, one per line."
    )
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = res.choices[0].message.content.strip().split("\n")
    return [l.strip().strip('"') for l in lines if l.strip()][:10]


def rank_with_model(candidates):
    vectors = embedder.encode(candidates)
    scores  = xgb_model.predict(vectors)
    ranked  = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    return ranked


def trained_llm_rank(idea):
    prompt = (
        "You are a YouTube title expert trained on millions of videos. "
        f"Generate 10 YouTube titles for: '{idea}'. "
        "Rank them from highest to lowest predicted click-through rate. "
        "No numbers or bullets, just titles in order, one per line."
    )
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    lines = res.choices[0].message.content.strip().split("\n")
    return [l.strip().strip('"') for l in lines if l.strip()][:10]


def search_youtube(query, max_results=3):
    try:
        search_res = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results + 3,
            order="viewCount"
        ).execute()

        video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]
        if not video_ids:
            return []

        stats_res = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids)
        ).execute()

        results = []
        for item in stats_res.get("items", []):
            if item["snippet"].get("categoryId") == MUSIC_CATEGORY:
                continue
            results.append({
                "title": item["snippet"]["title"],
                "views": int(item["statistics"].get("viewCount", 0)),
                "url":   f"https://www.youtube.com/watch?v={item['id']}",
                "channel": item["snippet"]["channelTitle"]
            })

        return sorted(results, key=lambda x: x["views"], reverse=True)[:max_results]
    except Exception as e:
        return []


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    idea = (request.get_json() or {}).get("idea", "").strip()
    if not idea:
        return jsonify({"error": "No idea provided"}), 400

    try:
        candidates = brainstorm(idea)
        ranked     = rank_with_model(candidates)
        top_title  = ranked[0][0]

        rationale = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Explain in one punchy sentence why '{top_title}' is high-CTR."}]
        ).choices[0].message.content.strip()

        return jsonify({
            "leaderboard": [{"title": t, "score": round(s, 2)} for t, s in ranked],
            "insight": rationale
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    idea = (request.get_json() or {}).get("idea", "").strip()
    if not idea:
        return jsonify({"error": "No idea provided"}), 400

    try:
        candidates   = brainstorm(idea)
        model_ranked = rank_with_model(candidates)
        ctr_ranked   = trained_llm_rank(idea)

        return jsonify({
            "llm_raw":     [{"title": t, "score": None} for t in candidates],
            "model_ranked":[{"title": t, "score": round(s, 2)} for t, s in model_ranked],
            "trained_llm": [{"title": t, "score": None} for t in ctr_ranked]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/groundtruth", methods=["POST"])
def groundtruth():
    title = (request.get_json() or {}).get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400

    try:
        results = search_youtube(title, max_results=3)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
