#we created this file specifically to load the model outside views to improve performace and loading speed as ram is laready utilised here

import os
import re
import torch
import numpy as np
import pickle
from django.apps import apps
from django.conf import settings
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Note: You can remove the manual MODEL_PATH/load_model logic here 
# since you are now fetching them from app_config (the better way).

def predict_toxicity(text):
    app_config = apps.get_app_config('userapp')
    bilstm_model = app_config.model
    bilstm_tokenizer = app_config.tokenizer
    bert_model = app_config.bert_model
    bert_tokenizer = app_config.bert_tokenizer

    if any(m is None for m in [bilstm_model, bilstm_tokenizer, bert_model, bert_tokenizer]):
        return {
            'ui_label': "Error",
            'ui_score': 0,
            'error': "Models not loaded correctly"
        }

    text_lower = str(text).lower().strip()

    # --- ENGINE 1: BiLSTM ---
    sequences = bilstm_tokenizer.texts_to_sequences([text_lower])
    padded = pad_sequences(sequences, maxlen=200, padding='pre')
    bilstm_prediction = bilstm_model.predict(padded, verbose=0)[0]
    
    cats = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    bilstm_scores = {cats[i]: float(bilstm_prediction[i]) for i in range(len(cats))}
    bilstm_max = max(bilstm_scores.values())

    # --- ENGINE 2: BERT ---
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        bert_probs = torch.sigmoid(outputs.logits).squeeze().tolist()
        bert_toxic_val = bert_probs[0]
        bert_identity_val = bert_probs[5]

    # --- ENSEMBLE LOGIC ---
    weighted_score = (bert_toxic_val * 0.7) + (bilstm_max * 0.3)

    # Contextual override for accuracy
    if bert_toxic_val < 0.10:
        verdict_score = bert_toxic_val
    else:
        verdict_score = weighted_score

    ui_score = round(verdict_score * 100, 2)
    ui_label = "Toxic" if verdict_score >= 0.50 else "Safe"

    return {
        'ui_label': ui_label,
        'ui_score': ui_score,
        'bilstm_raw': round(bilstm_max * 100, 2),
        'bert_raw': round(bert_toxic_val * 100, 2),
        'toxic': round(bilstm_scores['toxic'] * 100, 2),
        'severe_toxic': round(bilstm_scores['severe_toxic'] * 100, 2),
        'obscene': round(bilstm_scores['obscene'] * 100, 2),
        'threat': round(bilstm_scores['threat'] * 100, 2),
        'insult': round(bilstm_scores['insult'] * 100, 2),
        'identity_hate': round(max(bilstm_scores['identity_hate'], bert_identity_val) * 100, 2),
    }