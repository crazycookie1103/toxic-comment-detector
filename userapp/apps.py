import os
import pickle
import tensorflow as tf
from django.apps import AppConfig
# NEW: Import for BERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class UserappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'userapp'
    
    # Existing BiLSTM slots
    model = None
    tokenizer = None

    # NEW: BERT slots
    bert_model = None
    bert_tokenizer = None

    def ready(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # --- 1. Load your existing BiLSTM ---
        model_path = os.path.join(base_path, 'saved_models', 'bilstmfinal.keras')
        tokenizer_path = os.path.join(base_path, 'saved_models', 'tokenizerfinal.pickle')

        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                UserappConfig.model = tf.keras.models.load_model(model_path)
                with open(tokenizer_path, 'rb') as handle:
                    UserappConfig.tokenizer = pickle.load(handle)
                print("BiLSTM Loaded!")
            except Exception as e:
                print(f" ERROR LOADING BiLSTM: {e}")

        # --- 2. Load the BERT Transformer ---
        # We use 'unitary/toxic-bert' because it is pre-trained for toxicity
        try:
            print("🚀 Loading BERT Context Engine (Hugging Face)...")
            bert_name = "unitary/toxic-bert"
            UserappConfig.bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
            UserappConfig.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_name)
            print("BERT Transformer Loaded!")
        except Exception as e:
            print(f" ERROR LOADING BERT: {e}")