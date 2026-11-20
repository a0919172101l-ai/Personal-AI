
import streamlit as st
import torch

from model import PersonalAI


# 1. Website Settings
st.set_page_config(
    page_title="My Personal AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My Personal AI")
st.write("Ask me questions to learn more about me!")


# 2. Load Trained Model
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

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, vocab, checkpoint


model, vocab, checkpoint = load_model()

reverse_vocab = {
    value: key
    for key, value in vocab.items()
}


# 3. Generate Answer
def generate_answer(question):

    PAD = vocab["<PAD>"]
    SOS = vocab["<SOS>"]
    EOS = vocab["<EOS>"]

    max_q = checkpoint["max_q"]
    max_a = checkpoint["max_a"]

    # Convert question into numbers
    question_ids = [
        vocab[char]
        for char in question
        if char in vocab
    ]

    question_ids.append(EOS)

    question_ids = question_ids[:max_q]

    question_ids += (
        [PAD] * (max_q - len(question_ids))
    )

    question_tensor = torch.tensor(
        [question_ids],
        dtype=torch.long
    )

    # Generate answer
    answer_ids = [SOS]

    with torch.no_grad():

        question_embedded = model.embedding(
            question_tensor
        )

        _, hidden = model.encoder(
            question_embedded
        )

        for _ in range(max_a):

            input_tensor = torch.tensor(
                [[answer_ids[-1]]],
                dtype=torch.long
            )

            embedded = model.embedding(
                input_tensor
            )

            output, hidden = model.decoder(
                embedded,
                hidden
            )

            logits = model.output_layer(
                output[:, -1, :]
            )

            next_id = logits.argmax(
                dim=-1
            ).item()

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


# 4. Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# 5. Chat Interface
user_question = st.chat_input(
    "Ask me something..."
)

if user_question:

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.write(user_question)

    answer = generate_answer(user_question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)
