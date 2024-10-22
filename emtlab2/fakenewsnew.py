import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import string

# Load datasets
df_fake = pd.read_csv('emtlab2\Fake.csv')
df_true = pd.read_csv('emtlab2\True.csv')

# Assigning classes
df_fake['class'] = 0
df_true['class'] = 1

# Combine datasets
df = pd.concat([df_fake, df_true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.25, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Model Training and Evaluation

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
lr_predictions = lr_model.predict(x_test)
print("Logistic Regression Performance")
print(classification_report(y_test, lr_predictions))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
print("Decision Tree Performance")
print(classification_report(y_test, dt_predictions))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)
print("Random Forest Performance")
print(classification_report(y_test, rf_predictions))

# Gradient Boosting Classifier
gbc_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.15, max_depth=7, random_state=42, max_features='sqrt', subsample=0.8)
gbc_model.fit(x_train, y_train)
gbc_predictions = gbc_model.predict(x_test)
print("Gradient Boosting Classifier Performance")
print(classification_report(y_test, gbc_predictions))