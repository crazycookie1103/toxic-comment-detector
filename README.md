# Toxicity Detection Dashboard

A machine learning-powered web application designed to identify and classify toxic language in text datasets. This project implements a comparative framework between a state-of-the-art BERT Transformer and a high-efficiency BiLSTM architecture, using an ensemble method to provide finalized predictions.

## Tech Stack
* **Backend:** Django (Python)
* **Frontend:** HTML5, CSS3, JavaScript
* **Machine Learning:** TensorFlow, Keras, Hugging Face Transformers
* **Word Embeddings:** GloVe (Global Vectors for Word Representation)
* **Database:** SQLite

---

## Web Application Architecture
The web layer is built using the Django framework to provide a secure and functional environment for interacting with the machine learning models.

### 1. Authentication System
* **User Registration:** Provides a secure interface for new users to create accounts.
* **Session Management:** Utilizes Django's built-in authentication system for secure login and logout functionality, ensuring that only authorized users can access the analysis dashboard.

### 2. Analysis Dashboard
* **Text Processing Interface:** A dedicated dashboard where users can input text for evaluation.
* **Inference Engine:** Upon submission, the backend triggers the logic in `utils.py`, which loads the pre-trained models from the `ml 1/` directory.
* **Result Visualization:** Displays individual confidence scores from both BERT and BiLSTM alongside the finalized ensemble prediction.

---

## Machine Learning Pipeline

### 1. Data Preprocessing
Data is cleaned and structured within the `ml 1/` development environment to ensure optimal model performance:
* **Text Cleaning:** Removal of noise, including HTML tags, URLs, and non-alphanumeric characters.
* **Tokenization:**
   * **BiLSTM:** Employs standard word-level tokenization mapped to GloVe embeddings.
* **Sequence Standardization:** Inputs are padded or truncated to a fixed length to maintain consistent tensor dimensions.



### 2. Model Comparison and Ensemble Logic
The system implements a Weighted Soft-Voting Ensemble in `utils.py` to leverage the unique strengths of both architectures:

* **BERT (Transformer):** Serves as the primary contextual engine. By analyzing text bidirectionally, it identifies subtle toxic nuances that traditional models may overlook.
* **BiLSTM (RNN):** A Bidirectional Long Short-Term Memory network that captures long-term dependencies in text sequences. It acts as a robust secondary validator.
* **Ensemble Strategy:** The final toxicity classification is determined by a weighted average:
    **Final Score = (0.7 × BERT) + (0.3 × BiLSTM)**
    This weighting prioritizes BERT’s high contextual precision while using the BiLSTM to stabilize the overall prediction.



---

## Project Structure
* **`ml 1/`**: Research and development directory containing Jupyter Notebooks and saved model weights (.h5 or .keras).
* **`toxic_system/`**: Core Django project configuration, including settings and primary URL routing.
* **`userapp/`**: The main Django application handling the web layer and frontend templates.
    * `templates/`: Contains `dashboard.html`, `login.html`, and `register.html`.
* **`manage.py`**: Command-line utility for server management and database migrations.

---

## Data Acquisition
Due to file size constraints, external assets must be added manually to the `ml 1/data/` directory:
* **GloVe Embeddings:** [Download glove.6B.300d.txt from Stanford NLP](https://nlp.stanford.edu/projects/glove/)
* **Dataset:** [Toxic Comment Classification Challenge on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## Execution
To initialize the development server locally, execute the following command:
```bash
python manage.py runserver
