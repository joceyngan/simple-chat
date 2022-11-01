import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
# specify GPU
device = torch.device('cuda')

# We have prepared a chitchat dataset with 5 labels
df = pd.read_excel('./chitchat.xlsx')
df.head()
df['label'].value_counts()

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
# check class distribution
df['label'].value_counts(normalize = True)

# In this example we have used all the utterances for training purpose
train_text, train_labels = df['text'], df['label']