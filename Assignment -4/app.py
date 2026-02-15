import os
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel, BertConfig

st.set_page_config(page_title="Do You Agree?", layout="centered")

st.title("Do You Agree?")
st.write("Check whether two sentences agree, contradict, or are neutral.")

# Confirmed SNLI label mapping from your notebook
LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# Local folder next to app.py (must contain tokenizer files + sbert_nli.pt)
MODEL_DIR = "models/sbert_nli"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "sbert_nli.pt")

# Use the same max length you used for NLI tokenization
MAX_LEN = 64


def build_config_from_state(state_dict):
    # Infer core sizes from checkpoint tensors
    vocab_size, hidden_size = state_dict["encoder.embeddings.word_embeddings.weight"].shape
    max_pos, _ = state_dict["encoder.embeddings.position_embeddings.weight"].shape
    intermediate_size = state_dict["encoder.encoder.layer.0.intermediate.dense.weight"].shape[0]

    # Infer number of layers
    layer_keys = [
        k for k in state_dict.keys()
        if k.startswith("encoder.encoder.layer.")
        and k.endswith(".attention.self.query.weight")
    ]
    num_hidden_layers = len(layer_keys)

    # Infer attention heads (common head_dim=64; fallback to 4)
    num_attention_heads = max(1, hidden_size // 64)
    if hidden_size % num_attention_heads != 0:
        num_attention_heads = 4

    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_pos,
        type_vocab_size=2,
    )
    return cfg, hidden_size


class SBERTNLI(nn.Module):
    """
    Matches your training checkpoint style:
      - encoder.* keys for BERT
      - fc.* keys for classifier head
    """
    def __init__(self, config: BertConfig, hidden_size: int, num_labels: int = 3):
        super().__init__()
        # Build encoder from config to match the checkpoint architecture
        self.encoder = BertModel(config, add_pooling_layer=False)
        # Name it 'fc' because your checkpoint uses fc.weight and fc.bias
        self.fc = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, a, b):
        a_out = self.encoder(**a).last_hidden_state[:, 0]  # CLS (B, H)
        b_out = self.encoder(**b).last_hidden_state[:, 0]  # CLS (B, H)
        feats = torch.cat([a_out, b_out, torch.abs(a_out - b_out)], dim=1)  # (B, 3H)
        logits = self.fc(feats)
        return logits


@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Missing folder: {MODEL_DIR}")

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights file: {WEIGHTS_PATH}")

    # Load tokenizer saved from your notebook
    tok = BertTokenizerFast.from_pretrained(MODEL_DIR)

    # Load checkpoint
    raw_state = torch.load(WEIGHTS_PATH, map_location="cpu")

    # Drop pooler weights if they exist in checkpoint (safe)
    filtered_state = {k: v for k, v in raw_state.items() if not k.startswith("encoder.pooler.")}

    # Build a config that matches checkpoint shapes
    cfg, hidden_size = build_config_from_state(filtered_state)

    # Build model and load weights
    model = SBERTNLI(cfg, hidden_size)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    # Optional debugging in terminal logs
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return tok, model, device


# ---------- UI ----------
s1 = st.text_area("Sentence 1", placeholder="e.g., he is sleeping")
s2 = st.text_area("Sentence 2", placeholder="e.g., he is awake and running")

if st.button("Predict"):
    if not (s1.strip() and s2.strip()):
        st.warning("Please enter both sentences.")
    else:
        try:
            tokenizer, model, device = load_all()

            a = tokenizer(
                s1,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
            b = tokenizer(
                s2,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )

            a = {k: v.to(device) for k, v in a.items()}
            b = {k: v.to(device) for k, v in b.items()}

            with torch.no_grad():
                logits = model(a, b)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())

            st.subheader("Prediction")
            st.write(LABEL_MAP[pred])
            st.write(f"Confidence: {float(probs[pred]):.2f}")

            with st.expander("Show class probabilities"):
                st.write({LABEL_MAP[i]: float(probs[i]) for i in range(3)})

        except Exception as e:
            st.error("Model load or prediction failed.")
            st.code(str(e))
            st.info(
                "Check that you copied your Colab-saved folder into this project:\n"
                "models/sbert_nli/sbert_nli.pt and tokenizer files."
            )
