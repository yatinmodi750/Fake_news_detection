import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load the dataset
df = pd.read_csv('FakeNewsNet.csv', delimiter=',')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Applying the cleaning function to your dataset
df['title'] = df['title'].apply(clean_text)

print(df.columns)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['real'], test_size=0.2, random_state=42)
