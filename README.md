# Twitter Emotion Detection Transformer

A deep learning model to **detect emotions** from Twitter text, using:
- Specialized preprocessing for tweets
- Real-world emotion datasets
- A custom Transformer model built from scratch in PyTorch
- Pre-trained GloVe embeddings

---

## 📂 Project Structure

- **Data Preparation**: Download, clean, and balance the dataset.
- **Vocabulary Building**: Create a custom vocabulary using word frequencies and GloVe embeddings.
- **Sequence Conversion**: Transform cleaned texts into padded token sequences.
- **Dataset and DataLoader**: Build PyTorch Dataset and DataLoader for training/validation/testing.
- **Model Definition**: Implement an enhanced Transformer encoder.
- **Training Loop**: Train with weighted loss, learning rate scheduling, and early stopping.
- **Evaluation**: Report final test accuracy and inference on custom samples.

---

## 📥 Dataset

- **Source**: [Kaggle - Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
- **Format**: CSV file containing `content` (tweet text) and `sentiment` (emotion label).

---

## 🔍 Preprocessing

- **Demojization**: Convert emojis to text descriptions.
- **Cleaning**: Remove URLs, mentions, hashtags, non-alphabetic characters, and excessive repeated letters.
- **Tokenization**: Using `TweetTokenizer` (NLTK).
- **Normalization**:
  - Remove stopwords
  - Lemmatize and stem words
- **Label Filtering**: Keep only the top 6 most frequent emotions.
- **Oversampling**: Balance classes using RandomOverSampler.

---

## 🧠 Vocabulary

- Build a `EnhancedVocab` class with:
  - Special tokens: `[PAD]`, `[UNK]`, `[CLS]`
  - Mean and standard deviation of GloVe vectors for unknown word initialization
- Minimum word frequency: **2**
- Pre-trained Embeddings: **GloVe Twitter 100D**

---

## 🛠️ Model Architecture

**EnhancedTwitterTransformer**:
- **Embedding Layer**:
  - Initialized with GloVe
  - Fine-tunable
- **Positional Embedding**:
  - Trainable
- **Transformer Encoder**:
  - 4 layers
  - 4 attention heads
  - Hidden size: 256
  - GELU activation
  - Layer normalization first
- **Classifier Head**:
  - LayerNorm → Linear → GELU → Dropout → Linear

---

## ⚙️ Training Details

- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss (weighted by inverse sqrt of class counts)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 64
- **Epochs**: up to 50
- **Early Stopping**: Patience of 6 epochs
- **Gradient Clipping**: Max norm 0.5

---

## 📈 Evaluation Metrics

- Accuracy
- F1-Score (weighted)
- Precision
- Recall
- Full classification report

---

## 📊 Results

### Training Summary:

```
Train Loss: 0.5689 | Acc: 78.86%
Val Loss: 1.5684 | Acc: 60.67% | F1: 0.5972
```

### Classification Report (Validation):

| Label      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| neutral    | 0.4445    | 0.3711 | 0.4045   | 1296    |
| worry      | 0.4282    | 0.3449 | 0.3821   | 1296    |
| happiness  | 0.6070    | 0.6435 | 0.6247   | 1296    |
| sadness    | 0.5869    | 0.6543 | 0.6188   | 1296    |
| love       | 0.7169    | 0.7523 | 0.7342   | 1296    |
| surprise   | 0.7702    | 0.8742 | 0.8189   | 1296    |

- Overall Accuracy: **60.67%**
- Macro Avg F1: **0.5972**
- Weighted Avg F1: **0.5972**

**Early stopping after 6 epochs without improvement**

**Final Test Accuracy: 62.06%**

---

### Sample Inference Results:

```
Text: OMG just got tickets for the concert!!! 😍 #excited
Processed: omg got ticket concert smilingfacewithheartey excit
Predicted Emotion: happiness (0.67)
============================================================

Text: This service is terrible! Worst experience ever 😠
Processed: servic terribl worst experi ever angryfac
Predicted Emotion: worry (0.95)
============================================================

Text: Feeling so anxious about the interview tomorrow...
Processed: feel anxiou interview tomorrow
Predicted Emotion: worry (0.79)
============================================================

Text: Lost my pet today. I'm completely heartbroken 💔
Processed: lost pet today complet heartbroken brokenheart
Predicted Emotion: sadness (1.00)
============================================================

Text: What a beautiful morning! 🌞 #blessed
Processed: beauti morn sunwithfac bless
Predicted Emotion: happiness (0.48)
============================================================

Text: lol that's hilarious 😂
Processed: lol that' hilari facewithtearsofjoy
Predicted Emotion: neutral (0.42)
============================================================
```

---

## 📌 Requirements

- Python 3.7+
- PyTorch
- Scikit-learn
- Imbalanced-learn
- NLTK
- Pandas
- Numpy
- kagglehub
- emoji

```bash
pip install torch scikit-learn imbalanced-learn nltk pandas numpy kagglehub emoji
```

---

## 🚀 Notes

- Pre-trained GloVe embeddings are automatically downloaded if not available locally.
- The model supports fine-tuning embeddings during training.
- Balanced dataset improves emotion classification performance significantly.
- Data augmentation is applied during training to improve model generalization.
