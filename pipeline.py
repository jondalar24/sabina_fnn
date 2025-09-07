"""
Pipeline de preprocesamiento del texto
"""

import warnings
from tqdm import tqdm

warnings.simplefilter('ignore')
import time
from collections import OrderedDict
from typing import List, Iterable

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
from songs import SONG
from config import *
from torch.utils.data import DataLoader
from functools import partial



#1. Tokenizacion

def Tokenize_es(text: str)->List[str]:
    try:
        nltk.data.find('tokenizers/punkt/spanish.pickle')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    tokenizer = [tok.lower() for tok in word_tokenize(text, language='spanish') if tok.isalpha()]
    
    return tokenizer



#2. Indexación
def build_vocab_es(tokens):
    vocab = build_vocab_from_iterator([tokens], specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

#3. Embeddings
"""
No se utilizan de forma explícita, están integrados en el modelo,
pero se puede obtener visualizacion de los embeddings para obtener
pruebas didácticas
"""
def genembedding(vocab, dim= DIM_EMBEDDING):    
    vocab_size = len(vocab)
    embeddings = nn.Embedding(vocab_size, dim)
    return embeddings

#4. Generar pares Contexto-target
"""
Función que coge las "context" palabras previas a una objetivo
separando en tuplas de contexto-objetivo: los ngramas
"""
def genngrams(tokens):
    ngrams = [
        (
            [tokens[i - j - 1] for j in range(CONTEXT_SIZE)],
            tokens[i]
        )
        for i in range(CONTEXT_SIZE, len(tokens))
    ]
    return ngrams

#5. Carga de datos
def collate_batch(batch, vocab, device: str = 'cpu'):
    """
    Recibe un lote de pares (contexto, target) con tokens (str),
    los convierte a índices con vocab y devuelve tensores en el dispositivo correcto.
    """
    batch_size = len(batch)
    context_batch, target_batch = [], []


    for i in range(CONTEXT_SIZE, batch_size):
        target_batch.append(vocab([batch[i]]))
        context_batch.append(vocab([batch[i - j - 1] for j in range(CONTEXT_SIZE)]))
        

    return (
        torch.tensor(context_batch).to(device),
        torch.tensor(target_batch).to(device).reshape(-1)
    )

def padding(tokens):
    padding = BATCH_SIZE-len(tokens)%BATCH_SIZE
    tokens_pad = tokens + tokens[0:padding]
    return tokens_pad

def create_dataloader(tokens_pad,vocab, device, batch_size=BATCH_SIZE):
    custom_collate = partial(collate_batch, vocab=vocab, device=device)

    dataloader = DataLoader(
        tokens_pad, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate
    )
    return dataloader
