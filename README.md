
# [Paragraph_Semantic_Similarity_Analysis_App](https://paragraphsemanticsimilarityanalysisapp-6v59zwnbnfenxsgmahygf7.streamlit.app/)
![preview img](/semanticapp.png)

This application quantifies text similarity using semantic analysis. The tool measures how closely two paragraphs align in meaning, producing a similarity score between 0 and 1.

## Key Features
- Real-time semantic similarity scoring between two text inputs
- NLP preprocessing pipeline including tokenization, lemmatization, and stopword removal
- Sentence embedding using a pretrained transformer model (`all-mpnet-base-v2`)
- Cosine similarity calculation with MinMax scaling
- Graceful error handling via try-except for missing dependencies, empty inputs, and preprocessing failures

## Primary Applications
Semantic Textual Similarity (STS) assesses the degree to which two sentences are semantically equivalent, with use cases spanning:
- Information retrieval
- Text classification
- Recommendation systems
- Plagiarism detection

## Technical Implementation
The solution utilizes Streamlit for deployment, allowing users to input text pairs and receive real-time similarity predictions. The codebase consists of:
- `semantic_simil_app.py` — main application file with full try-except error handling
- `requirements.txt` — all Python dependencies
- `runtime.txt` — pins Python version to 3.11 for reliable deployment
- A Jupyter notebook for exploratory analysis and visualization

## Requirements
- Python 3.11 (see `runtime.txt`) — spaCy and thinc do not support Python 3.13+
- Dependencies listed in `requirements.txt`

## How to Run Locally
1. Clone the repository
   ```bash
   git clone https://github.com/ShilpaAjitheks/Paragraph_Semantic_Similarity_Analysis_App.git
   cd Paragraph_Semantic_Similarity_Analysis_App```
2. Create a virtual environment with Python 3.11
   ```python -m venv env
    env\Scripts\activate        # Windows
    source env/bin/activate     # Mac/Linux```
3. Install dependencies
    ```pip install -r requirements.txt
    python -m spacy download en_core_web_sm ```
4. Run the app
   ```streamlit run semantic_simil_app.py```
   
