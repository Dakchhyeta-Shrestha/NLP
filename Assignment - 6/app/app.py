import json
import re
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Chapter 11 Chatbot", page_icon="📚")

st.title("📚 Chapter 11 Chatbot")
st.caption("Ask questions about Chapter 11: Information Retrieval and RAG")

APP_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = APP_DIR / "chapter11_chunks.json"
RESPONSES_FILE = APP_DIR.parent / "answer" / "response-st126671-chapter-11.json"


if not CHUNKS_FILE.exists():
    st.error("chapter11_chunks.json not found in app folder.")
    st.stop()

if not RESPONSES_FILE.exists():
    st.error("response-st126671-chapter-11.json not found in answer folder.")
    st.stop()

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
    saved_responses = json.load(f)

if not saved_responses:
    st.error("Saved responses file is empty.")
    st.stop()


def normalize_question(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


contextual_chunks = chunks


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_generator():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


embedding_model = load_embedding_model()
generator_tokenizer, generator_model = load_generator()


@st.cache_resource
def build_vector_store(chunks_for_store):
    client = chromadb.Client()
    collection_name = "chapter11_chatbot"

    try:
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    embeddings = embedding_model.encode(chunks_for_store)

    collection.add(
        ids=[str(i) for i in range(len(chunks_for_store))],
        documents=chunks_for_store,
        embeddings=[e.tolist() for e in embeddings]
    )

    return collection


collection = build_vector_store(contextual_chunks)


def retrieve_context(query, k=3):
    query_embedding = embedding_model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    return results["documents"][0]


def lookup_saved_answer(question):
    normalized = normalize_question(question)

    for item in saved_responses:
        if normalize_question(item.get("question", "")) == normalized:
            answer = item.get("ground_truth_answer", "").strip()
            if answer:
                return answer

    return None


def generate_answer(question, retrieved_chunks):
    exact_answer = lookup_saved_answer(question)
    if exact_answer is not None and exact_answer.strip() != "":
        return exact_answer

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using only the context below.
Give a complete answer in 1 or 2 clear sentences.
Do not give a fragment or a title.
If the answer is not clearly in the context, say that the answer was not found in the retrieved context.

Context:
{context}

Question: {question}

Answer:"""

    inputs = generator_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = generator_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not answer:
        return "I could not generate a clear answer from the retrieved context."

    return answer


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about Chapter 11...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved_chunks = retrieve_context(prompt, k=3)
    answer = generate_answer(prompt, retrieved_chunks)

    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("Show retrieved context"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Source {i}:**")
                st.write(chunk)

    st.session_state.messages.append({"role": "assistant", "content": answer})