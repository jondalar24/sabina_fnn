import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import NGramLanguageModeler
from pipeline import Tokenize_es, build_vocab_es, create_dataloader, genngrams, collate_batch, padding
from train import predict_next_word, train_model, visualize_embeddings
from songs import SONG
from config import *  # Carga hiperparámetros como CONTEXT_SIZE, EMBEDDING_DIM, etc.

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1. Preparación del entorno
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------------------------
# 2. Tokenización e indexación
# ----------------------------
tokens = Tokenize_es(SONG)


vocab = build_vocab_es(tokens)
ngrams = genngrams(tokens)

# ----------------------------
# 3. DataLoader con collate
# ----------------------------
tokens_pad = padding(tokens)

dataloader = create_dataloader(tokens_pad, vocab=vocab, device=device)

# ----------------------------
# 4. Modelo, optimizador y pérdida
# ----------------------------
model = NGramLanguageModeler(len(vocab), DIM_EMBEDDING, CONTEXT_SIZE).to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# 5. Entrenamiento
# ----------------------------
losses = train_model(
    dataloader, model, SONG,
    optimizer, criterion,
    device,
    tokens,vocab,
    epochs=N_EPOCHS,
    show=SHOW_EVERY    
)

# ----------------------------
# 6. Curvas de entrenamiento
# ----------------------------

def smooth(y, window_size=5):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

def plot_curves(losses):
    plt.figure()
    plt.plot(smooth(losses), label='Cross Entropy Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Pérdida por época')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    ppl = np.exp(np.array(losses))
    plt.plot(smooth(ppl), label='Perplejidad')
    plt.xlabel('Épocas')
    plt.ylabel('Perplexity')
    plt.title('Perplejidad por época')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_curves(losses)

# ----------------------------
# 7. Visualización de embeddings
# ----------------------------
visualize_embeddings(model, vocab)

# ----------------------------
# 8. Futuras mejoras
# ----------------------------

print("Espero que les haya gustado la nueva canción de sabina...")