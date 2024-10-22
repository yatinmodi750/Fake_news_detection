import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from bs4 import BeautifulSoup
from googlesearch import search
import requests
# Load the dataset
df = pd.read_csv('FakeNewsNet.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['real'], test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data
X_train_vectors = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_vectors = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vectors, y_train)

# Accept headline input
headline = input("Enter a headline: ")

# Search the internet for the headline
search_results = search(headline, num_results=1)

# Extract the first search result
first_result = next(search_results)

# Scrape the webpage content
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
response = requests.get(first_result, headers=headers)
webpage = BeautifulSoup(response.content, 'html.parser')
webpage_text = webpage.get_text()

# Transform the scraped text
headline_vector = vectorizer.transform([webpage_text])

# Predict the probabilities of each class label
probabilities = classifier.predict_proba(headline_vector)

# Print the probabilities
print("Probability of being false:", probabilities[0][0])
print("Probability of being true:", probabilities[0][1])

# Print the prediction based on the highest probability
prediction = classifier.predict(headline_vector)
if prediction[0] == 1:
    print("The headline is predicted to be true.")
else:
    print("The headline is predicted to be false.")
