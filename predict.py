import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from data_preprocessing import clean_email
from config import DEVICE, MAX_LENGTH


def _tokenize(text, tokenizer, overflow=False):
    kwargs = dict(return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    if overflow:
        kwargs |= dict(stride=64, return_overflowing_tokens=True)
    inputs = tokenizer(text, **kwargs)
    return {k: v.to(DEVICE) for k, v in inputs.items() if k != "overflow_to_sample_mapping"}


def _forward(inputs, model, **kwargs):
    with torch.no_grad():
        model.eval()
        return model(**inputs, **kwargs)


def _print_header(text, probs, width=100):
    print("=" * 50)
    print("TEXT:", text[:width], "...")
    print(f"Normal: {round(probs[0] * 100, 2)}%  |  Phishing: {round(probs[1] * 100, 2)}%")


def predict_proba(text, model, tokenizer):
    inputs = _tokenize(text, tokenizer, overflow=True)
    outputs = _forward(inputs, model)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
    return np.mean(probs, axis=0)


def predict_proba_raw(text, model, tokenizer):
    return predict_proba(clean_email(text), model, tokenizer)


def _lime_predict(texts, model, tokenizer):
    return np.array([predict_proba_raw(t, model, tokenizer) for t in texts])


def explain_prediction(text, model, tokenizer):
    probs = predict_proba_raw(text, model, tokenizer)
    _print_header(text, probs, width=150)

    explainer = LimeTextExplainer(class_names=["normal", "phishing"])
    exp = explainer.explain_instance(
        text,
        lambda t: _lime_predict(t, model, tokenizer),
        num_features=5,
        num_samples=500,
    )

    suspicious = [(w, s) for w, s in exp.as_list() if s > 0]
    print("\nPodejrzane słowa (LIME):")
    if not suspicious:
        print("Brak.")
    else:
        for word, score in suspicious:
            print(f"  [ {word} ] -> {round(score, 3)}")


def explain_attention(text, model, tokenizer):
    inputs = _tokenize(clean_email(text), tokenizer, overflow=True)
    outputs = _forward(inputs, model, output_attentions=True)

    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
    probs = np.mean(probs, axis=0)

    attentions = outputs.attentions[-1]
    cls_attn = attentions[:, :, 0, :].mean(dim=(0, 1)).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    SKIP = {"[CLS]", "[SEP]", "[PAD]"}
    token_scores = sorted(
        ((t, s) for t, s in zip(tokens, cls_attn) if t not in SKIP),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n")
    _print_header(text, probs)
    print("\nNajbardziej podejrzane fragmenty (Attention):")
    for token, score in token_scores[:10]:
        print(f"  [ {token} ] -> {round(float(score), 3)}")
