# Neural Collaborative Filtering (NCF) Recommender

PyTorch implementation of Neural Collaborative Filtering (NCF/NeuMF) combining Generalized Matrix Factorization (GMF) and Deep Multi-Layer Perceptron (MLP) branches on the MovieLens-100K benchmark.

## Benchmark Results
- **Hit Rate@10 (HR@10):** 0.3924
- **NDCG@10:** 0.2170
- **Evaluation Protocol:** Leave-One-Out against 99 negative items per user.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py