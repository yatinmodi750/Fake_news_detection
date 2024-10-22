import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# Include your preprocessing function here
def wordopt(text): ...

# Load and preprocess your data
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
# Concatenate and preprocess steps as before, up to the vectorization
vectorization = TfidfVectorizer()
vectorized_data = vectorization.fit_transform(df["text"])
# Save the preprocessed data and the vectorizer for later use
joblib.dump(vectorized_data, 'vectorized_data.pkl')
joblib.dump(vectorization, 'vectorizer.pkl')
