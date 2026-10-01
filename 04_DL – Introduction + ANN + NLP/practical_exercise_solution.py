"""
Topic 04 Practical Exercise Reference Solution
Task: "The Customer Review Sentiment & Intent Intelligence Engine"
"""

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec

# ---------------------------------------------------------
# Step 1: Generate Synthetic Dataset of Customer Reviews
# ---------------------------------------------------------
raw_data = [
    ("The neural network model trained with GPU yields outstanding accuracy!", 1),
    ("Poor customer service, terrible build quality and delayed shipping!", 0),
    ("Deep learning algorithms require normalized datasets and hyperparameter tuning.", 1),
    ("Frustrating experience. The product broke down within two days of delivery.", 0),
    ("Excellent performance and crisp high resolution display. Highly recommended!", 1),
    ("Horrible purchase. The battery drains fast and customer care was unhelpful.", 0),
    ("Artificial neural networks with batch normalization converge extremely fast.", 1),
    ("Faulty device, defective screen, and completely useless technical support.", 0),
    ("State of the art NLP embeddings captured semantic context remarkably well.", 1),
    ("Worst product ever! Total waste of money and horrible user interface.", 0),
    ("The Word2Vec skip-gram model mapped similar words close in vector space.", 1),
    ("Defective item received. Package was damaged and refund was denied.", 0),
] * 25  # Expand dataset size to 300 samples

df = pd.DataFrame(raw_data, columns=['review_text', 'sentiment'])

# ---------------------------------------------------------
# Step 2: NLP Preprocessing
# ---------------------------------------------------------
SIMPLE_STOPWORDS = {'the', 'a', 'an', 'and', 'is', 'was', 'of', 'in', 'to', 'with', 'for', 'on', 'this', 'that', 'it'}

def clean_text(text):
    text = text.lower()                                  # Lowercasing
    text = re.sub(r'[^a-zA-Z\s]', '', text)             # Punctuation & number removal
    tokens = text.split()
    tokens = [t for t in tokens if t not in SIMPLE_STOPWORDS] # Stopword filtering
    return tokens

df['tokens'] = df['review_text'].apply(clean_text)
df['cleaned_text'] = df['tokens'].apply(lambda x: " ".join(x))

# ---------------------------------------------------------
# Step 3: Feature Extraction
# 1) TF-IDF Vectorization
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment'].values

# 2) Word2Vec Embeddings
w2v_model = Word2Vec(sentences=df['tokens'], vector_size=50, window=3, min_count=1, sg=1)

def get_doc_vector(tokens):
    vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vecs) == 0:
        return np.zeros(50)
    return np.mean(vecs, axis=0)

X_w2v = np.array([get_doc_vector(t) for t in df['tokens']])

# ---------------------------------------------------------
# Step 4: Define Deep Neural Network with Regularization
# ---------------------------------------------------------
class SentimentANN(nn.Module):
    def __init__(self, input_dim):
        super(SentimentANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),          # Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.3),              # Dropout (30%)
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),              # Dropout (20%)
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

# Train-Test Split for TF-IDF features
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Instantiate Model, Loss, Optimizer
model = SentimentANN(input_dim=X_tfidf.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4) # L2 Regularization

# ---------------------------------------------------------
# Step 5: Training Loop with Early Stopping
# ---------------------------------------------------------
patience = 5
best_loss = float('inf')
patience_counter = 0

print("--- Training Artificial Neural Network (TF-IDF Features) ---")
for epoch in range(1, 40):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    val_loss /= len(val_loader.dataset)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    # Early Stopping Check
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_sentiment_ann.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch}!")
            break

# ---------------------------------------------------------
# Step 6: Evaluation
# ---------------------------------------------------------
model.load_state_dict(torch.load("best_sentiment_ann.pt"))
model.eval()
with torch.no_grad():
    val_preds_prob = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    val_preds = (val_preds_prob >= 0.5).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_val, val_preds, target_names=['Negative', 'Positive']))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))
