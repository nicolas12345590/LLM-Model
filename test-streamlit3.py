import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json
import os

st.set_page_config(page_title="News Empfehlungssystem", layout="wide")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModel.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_hidden_states=True,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden.mean(dim=1)
    return embedding.cpu().squeeze()

def build_user_history_text(articles_with_feedback):
    return " ".join([
        f"User {'liked' if liked else 'disliked'} article {i+1}: \"{text}\""
        for i, (text, liked) in enumerate(articles_with_feedback)
    ])

def get_user_embedding(user_articles):
    text = build_user_history_text(user_articles)
    return get_embedding(text)

USER_FILE = "user_data.json"

def load_user_data():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    else:
        return []

def save_user_data(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f)

ALL_ARTICLES = [
    "OpenAI launches new GPT model with improved reasoning capabilities.",
    "Scientists discover potential cure for certain types of cancer.",
    "NASA plans manned mission to Mars within the next decade.",
    "Economy sees strong recovery after inflation rates stabilize.",
    "Climate change causes extreme weather events in Europe.",
    "Apple unveils AR headset at Worldwide Developers Conference.",
    "Google integrates advanced AI tools into Google Docs.",
    "Tesla introduces self-driving features in latest software update.",
    "Germany wins the World Cup in dramatic final match.",
    "New study shows that daily walking reduces risk of heart disease."
]

st.title("📰 News Empfehlungssystem")

# Lade oder initialisiere Userdaten
user_data = load_user_data()
if not user_data:
    st.info("📌 Erste Session: Wir starten mit Beispiel-Userdaten.")
    user_data = [
        ("OpenAI launches new GPT model with improved reasoning capabilities.", True),
        ("Tesla introduces self-driving features in latest software update.", True),
        ("Climate change causes extreme weather events in Europe.", False)
    ]
    save_user_data(user_data)

# Session State für temporäre Bewertung ohne reload
if "temp_user_data" not in st.session_state:
    st.session_state.temp_user_data = user_data.copy()

if "recommendation_shown" not in st.session_state:
    st.session_state.recommendation_shown = False

if "rated_articles" not in st.session_state:
    st.session_state.rated_articles = set(article for article, _ in st.session_state.temp_user_data)

# Funktion zum Hinzufügen von Feedback in session_state (lokal)
def add_feedback(article, liked):
    if article not in st.session_state.rated_articles:
        st.session_state.temp_user_data.append((article, liked))
        st.session_state.rated_articles.add(article)

# Button zum Aktualisieren des Nutzerprofils und speichern
if st.button("🔄 Nutzerprofil aktualisieren und Empfehlungen berechnen"):
    save_user_data(st.session_state.temp_user_data)
    st.session_state.recommendation_shown = True

if st.session_state.recommendation_shown:
    user_emb = get_user_embedding(st.session_state.temp_user_data)
    already_rated = [entry[0] for entry in st.session_state.temp_user_data]
    remaining = [a for a in ALL_ARTICLES if a not in already_rated]

    if not remaining:
        st.success("✅ Du hast bereits alle Artikel bewertet!")
    else:
        article_embeddings = [get_embedding(a) for a in remaining]
        similarities = [F.cosine_similarity(user_emb, emb, dim=0).item() for emb in article_embeddings]
        top_idx = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]

        st.subheader("🔍 Deine 3 besten Empfehlungen:")

        for i in top_idx:
            article = remaining[i]
            with st.expander(f"🔸 Artikel: {article[:60]}..."):
                # Farbige Markierung wenn bereits bewertet (lokal)
                liked_flag = None
                for art, liked in st.session_state.temp_user_data:
                    if art == article:
                        liked_flag = liked
                        break
                bg_color = ""
                if liked_flag is True:
                    bg_color = "background-color: #d4edda;"  # grün
                elif liked_flag is False:
                    bg_color = "background-color: #f8d7da;"  # rot

                st.markdown(f"<div style='{bg_color} padding: 10px; border-radius: 5px;'>{article}</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"👍 Gefällt mir {i}", key=f"like_{article}"):
                        add_feedback(article, True)
                with col2:
                    if st.button(f"👎 Gefällt mir nicht {i}", key=f"dislike_{article}"):
                        add_feedback(article, False)
else:
    st.info("📥 Bitte aktualisiere zuerst dein Nutzerprofil, um Empfehlungen zu erhalten.")

# Verlauf anzeigen & Reset
st.sidebar.header("📚 Bewertungsverlauf")
for i, (text, liked) in enumerate(st.session_state.temp_user_data):
    emoji = "👍" if liked else "👎"
    color = "#d4edda" if liked else "#f8d7da"
    st.sidebar.markdown(f"<div style='background-color:{color}; padding:5px; border-radius:5px'>{emoji} {text[:60]}...</div>", unsafe_allow_html=True)

if st.sidebar.button("🗑️ Verlauf zurücksetzen"):
    if os.path.exists(USER_FILE):
        os.remove(USER_FILE)
    st.session_state.temp_user_data = []
    st.session_state.recommendation_shown = False
    st.session_state.rated_articles = set()
    st.experimental_rerun()

