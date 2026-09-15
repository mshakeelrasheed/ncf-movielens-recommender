# 🎬 Neural Collaborative Filtering (NCF) Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end implementation of the **Neural Collaborative Filtering (NCF/NeuMF)** framework based on the landmark research paper:

> *Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua. "Neural Collaborative Filtering." In Proceedings of the 26th International Conference on World Wide Web (WWW '17).*

The system models non-linear latent user-item feature interactions on implicit interaction data, evaluates recommendation quality using ranking metrics, and provides an interactive Streamlit interface for real-time recommendations.

---

## 📌 Architecture Overview

Traditional Matrix Factorization (MF) models interactions via a linear dot product of user and item latent vectors:

$$\hat{y}_{ui} = p_u^T q_i$$

NCF addresses the limitations of linear interaction modeling by introducing **Neural Matrix Factorization (NeuMF)**, which combines:

1. **Generalized Matrix Factorization (GMF)**  
   Computes an element-wise product of user and item embeddings to capture linear interaction patterns.

2. **Multi-Layer Perceptron (MLP)**  
   Concatenates user and item embeddings and passes them through non-linear neural network layers to learn more complex interaction patterns.

### Architecture

```text
                 Input
            (User ID, Item ID)
                    |
          +---------+---------+
          |                   |
          v                   v
     GMF Branch           MLP Branch
          |                   |
 User Embedding        User Embedding
 Item Embedding        Item Embedding
          |                   |
          v                   v
 Element-wise          Concatenation
 Product                    |
          |                   v
          |               MLP Layers
          |                   |
          +---------+---------+
                    |
              Concatenation
                    |
             Linear Projection
                    |
                 Sigmoid
                    |
              Recommendation
```

---

## 📊 Dataset & Evaluation

The model is evaluated on the **MovieLens-100K** dataset:

- **943 users**
- **1,682 movies**
- **100,000 ratings**
- Ratings converted into binary implicit interactions
- Negative sampling used during training

### Evaluation Protocol

**Leave-One-Out (LOO):**

For each user:

- The latest recorded interaction is held out for testing.
- Earlier interactions are used for training.
- The positive test item is combined with randomly sampled unobserved negative items.
- The model ranks the candidate items.
- Top-K recommendation quality is measured.

### Metrics

| Metric | Description |
|---|---|
| **Hit Rate@10 (HR@10)** | Measures whether the ground-truth item appears among the top 10 recommendations. |
| **NDCG@10** | Measures both whether the correct item is recommended and how highly it is ranked. |

> **Note:** Reported metric values should only be included after reproducing the evaluation with the current implementation. Do not claim benchmark scores that have not been independently verified.

---

## 🗂️ Project Structure

```text
ncf-movielens-recommender/
│
├── checkpoints/
│   └── ncf_model.pt          # Pre-trained NeuMF model weights
│
├── data/
│   └── u.item                # Movie catalog metadata & ID mappings
│
├── src/
│   ├── __init__.py
│   └── model.py              # PyTorch NeuMF model architecture
│
├── app.py                    # Interactive Streamlit inference dashboard
├── requirements.txt          # Project dependencies
└── README.md                 # Technical documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mshakeelrasheed/ncf-movielens-recommender.git
cd ncf-movielens-recommender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

The application provides an interactive interface for selecting a user and generating movie recommendations.

---

## 🧠 Model Configuration

Example training configuration:

| Parameter | Value |
|---|---|
| GMF Embedding Dimension | 32 |
| MLP Embedding Dimension | 32 |
| MLP Hidden Layers | `[64, 32, 16]` |
| Activation | ReLU |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 256 |

These values can be adjusted depending on the available compute resources and experimental requirements.

---

## 🔬 Why Neural Collaborative Filtering?

Traditional collaborative filtering methods often rely on simple linear interactions between users and items.

NCF uses neural networks to learn **non-linear user-item relationships**, making it possible to model more complex preference patterns.

The project demonstrates practical concepts including:

- User and item embeddings
- Collaborative filtering
- Neural recommendation models
- Negative sampling
- Ranking-based evaluation
- Top-K recommendation
- Deep learning model deployment

---

## 🎯 Example Workflow

```text
User
  |
  v
Select / Identify User
  |
  v
Learned User Embedding
  |
  +----------------------+
  |                      |
  v                      v
GMF Branch            MLP Branch
  |                      |
  +----------+-----------+
             |
             v
       Neural Prediction
             |
             v
       Score All Candidates
             |
             v
        Rank by Score
             |
             v
       Top-K Movies
```

---

## 💻 Technologies

- **Python**
- **PyTorch**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **MovieLens Dataset**

---

## 📚 Research Reference

This implementation is inspired by:

> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). **Neural Collaborative Filtering.** Proceedings of the 26th International Conference on World Wide Web (WWW '17).

The original work introduced neural network-based collaborative filtering approaches including **GMF, MLP, and NeuMF**.

---

## ⚠️ Reproducibility

For research or CV purposes, keep the following information documented:

- Dataset version
- Train/test split strategy
- Negative sampling ratio
- Random seed
- Model hyperparameters
- Number of training epochs
- Evaluation metrics
- Hardware used for training

This makes experimental results easier to reproduce and compare.

---

## 📈 Future Improvements

Potential extensions include:

- [ ] Compare NCF against Matrix Factorization
- [ ] Experiment with different embedding dimensions
- [ ] Add user/movie metadata
- [ ] Implement hybrid recommendation
- [ ] Add Precision@K and Recall@K
- [ ] Perform hyperparameter tuning
- [ ] Add explainable recommendations
- [ ] Deploy the model using Hugging Face Spaces
- [ ] Experiment with transformer-based recommenders

---

## 📜 License

Distributed under the MIT License.

---

## 👤 Author

**Muhammad Shakeel Rasheed**

GitHub: [@mshakeelrasheed](https://github.com/mshakeelrasheed)
