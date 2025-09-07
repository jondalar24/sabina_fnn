from __future__ import annotations
import argparse
import math
import os
import random
import re
import string
from collections import Counter
from typing import List, Iterable, Optional


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from config import *
from pipeline import Tokenize_es


def train_model(dataloader, model, song, optimizer, criterion, device, tokens, vocab, epochs=N_EPOCHS, show=SHOW_EVERY):
    """
    Entrena el modelo N-Gram durante un número de épocas.

    Parámetros:
    - dataloader: batches de datos (contexto, target)
    - model: instancia de NGramLanguageModeler
    - song: texto original para generar canciones
    - optimizer: optimizador (SGD, Adam, etc.)
    - criterion: función de pérdida (CrossEntropyLoss)
    - device: 'cpu' o 'cuda'
    - epochs: número de épocas
    - show: frecuencia con la que se genera texto durante entrenamiento

    Retorna:
    - Lista de pérdidas medias por época
    """
    MY_LOSS = []

    for epoch in tqdm(range(epochs), desc="Entrenando modelo"):
        total_loss = 0.0
        my_song = ""

        #model.train()

        for context, target in dataloader:
            model.zero_grad()
            predicted =  model(context)
            loss = criterion(predicted, target.reshape(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        

        if epoch % show == 0:
            selected_line = pickrandomline(song)
            my_song += write_song(model, selected_line, tokens, vocab, device)

            print(f"Epoch {epoch:3d} ")
            print("\nCanción generada:\n")
            print(my_song)

        MY_LOSS.append(total_loss / len(dataloader))

    return MY_LOSS

def visualize_embeddings(model, vocab, top_n: int = 100):
    print("\n[Visualizando embeddings con t-SNE]")
    emb_weights = model.embeddings.weight.cpu().detach().numpy()
    labels = [vocab.get_itos()[i] for i in range(top_n)]
    vectors = emb_weights[:top_n]


    tsne = TSNE(n_components=2, random_state=42, init='pca')
    X_2d = tsne.fit_transform(vectors)


    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = X_2d[i, 0], X_2d[i, 1]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), fontsize=9)
    plt.title("Visualización 2D de Embeddings (t-SNE)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def predict_next_word(model, input_text, vocab, device):
    """
    Predice la siguiente palabra dada una secuencia de entrada parcial.
    Si el número de palabras es menor que CONTEXT_SIZE, se rellena con tokens "<unk>".
    """
    index_to_token = vocab.get_itos()
    tokens = input_text.lower().split()

    if len(tokens) < CONTEXT_SIZE:
        tokens = ["<unk>"] * (CONTEXT_SIZE - len(tokens)) + tokens
    else:
        tokens = tokens[-CONTEXT_SIZE:]  # Solo las últimas n palabras

    context_idxs = torch.tensor(vocab(tokens), dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(context_idxs)
        predicted_idx = torch.argmax(output).item()
        predicted_word = index_to_token[predicted_idx]

    return predicted_word


def write_song(model, my_song, tokens, vocab, device, number_of_words=150):
    
    # Get the mapping from index to word for decoding predictions
    index_to_token = vocab.get_itos()

    # Loop to generate the desired number of words
    for i in range(number_of_words):

        with torch.no_grad():  # Disable gradient computation for inference
            
            # Prepare the input context by extracting the last CONTEXT_SIZE words from tokens
            context = torch.tensor(
                vocab([tokens[i - j - 1] for j in range(CONTEXT_SIZE)])
            ).to(device)  # Move to CPU/GPU as required
            
            # Predict the next word by selecting the word with the highest probability
            word_idx = torch.argmax(model(context))  # Get index of the most likely next word
            
            # Append the predicted word to the generated text
            my_song += " " + index_to_token[word_idx.detach().item()]

    return my_song  # Return the generated lyrics

    


def pickrandomline(song):    
    
    # Split the song into individual lines
    lines = song.split("\n")  
    
    # Randomly select a line and remove leading/trailing whitespace
    selected_line = random.choice(lines).strip()
    
    return selected_line  # Return the randomly selected line
