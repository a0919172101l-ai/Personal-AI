
import streamlit as st
import torch
import pandas as pd
import re
from difflib import SequenceMatcher

from model import PersonalAI


# 1. Website Settings
st.set_page_config(
    page_title="My Personal AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My Personal AI")
st.write("Ask me questions to learn more about me!")


# 2. Load Training Data
@st.cache_data
def load_data():
    data = pd.read_csv("training_data.csv")
    data = data.dropna(subset=["question", "answer"])
    return data


data = load_data()


# 3. Load Trained Model
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "model_weights.pth",
        map_location="cpu",
        weights_only=True
    )

    vocab = checkpoint["vocab"]

    model = PersonalAI(
        vocab_size=len(vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, vocab, checkpoint


model, vocab, checkpoint = load_model()

reverse_vocab = {
    value: key
    for key, value in vocab.items()
}


# 4. Normalize Questions
def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# 5. Search Personal Information
def search_data(question):
    normalized_question = normalize(question)

    stop_words = {
        "what", "is", "are", "my", "your", "the",
        "a", "an", "and", "or", "do", "does",
        "i", "you", "me", "tell", "about", "can",
        "please", "when", "who", "how", "much",
        "many", "to", "of", "in", "on", "with",
        "for", "be", "am", "was", "were"
    }

    question_words = set(normalized_question.split())
    keywords = question_words - stop_words

    matched_answers = []
    matched_questions = []

    for _, row in data.iterrows():
        stored_question = normalize(row["question"])

        stored_words = set(stored_question.split())
        stored_keywords = stored_words - stop_words

        if not stored_keywords:
            continue

        # Measure keyword similarity
        overlap = len(keywords.intersection(stored_keywords))

        score = overlap / len(stored_keywords)

        # Also check overall question similarity
        similarity = SequenceMatcher(
            None,
            normalized_question,
            stored_question
        ).ratio()

        if score >= 0.75 or similarity >= 0.82:
            if stored_question not in matched_questions:
                matched_questions.append(stored_question)
                matched_answers.append(row["answer"])

    if matched_answers:
        return "\n\n".join(matched_answers)

    return None


# 6. Generate Answer with GRU
def generate_answer(question):
    PAD = vocab["<PAD>"]
    SOS = vocab["<SOS>"]
    EOS = vocab["<EOS>"]

    max_q = checkpoint["max_q"]
    max_a = checkpoint["max_a"]

    question_ids = [
        vocab[char]
        for char in question
        if char in vocab
    ]

    question_ids.append(EOS)
    question_ids = question_ids[:max_q]

    question_ids += [PAD] * (max_q - len(question_ids))

    question_tensor = torch.tensor(
        [question_ids],
        dtype=torch.long
    )

    answer_ids = [SOS]

    with torch.no_grad():
        question_embedded = model.embedding(question_tensor)
        _, hidden = model.encoder(question_embedded)

        for _ in range(max_a):
            input_tensor = torch.tensor(
                [[answer_ids[-1]]],
                dtype=torch.long
            )

            embedded = model.embedding(input_tensor)

            output, hidden = model.decoder(
                embedded,
                hidden
            )

            logits = model.output_layer(output[:, -1, :])

            next_id = logits.argmax(dim=-1).item()

            if next_id in [PAD, SOS, EOS]:
                break

            answer_ids.append(next_id)

    answer = "".join(
        reverse_vocab[token]
        for token in answer_ids[1:]
    )

    if not answer:
        return "I could not generate an answer."

    return answer


# 7. Hybrid Answer System
def get_answer(question):
    database_answer = search_data(question)

    if database_answer is not None:
        return database_answer

    return generate_answer(question)


# 8. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# 9. Chat Interface
user_question = st.chat_input("Ask me something...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.write(user_question)

    answer = get_answer(user_question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)
