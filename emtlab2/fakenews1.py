import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # For saving and loading models
import sys  # For command-line arguments
import os  # For checking file existence

# Assume your CSV file names are stored in variables
fake_news_file = 'emtlab2\Fake.csv'
true_news_file = 'emtlab2\True.csv'
PREPROCESSED_FILE = 'preprocessed_data.pkl'
MODEL_FILE = 'trained_model.pkl'

def load_and_preprocess_data():
    if not os.path.exists(PREPROCESSED_FILE):
        fake_news = pd.read_csv(fake_news_file)
        true_news = pd.read_csv(true_news_file)

        fake_news['label'] = 'fake'
        true_news['label'] = 'true'

        data = pd.concat([fake_news, true_news], ignore_index=True)
        # Your preprocessing steps here
        preprocessed_data = data  # Placeholder for actual preprocessing
        joblib.dump(preprocessed_data, PREPROCESSED_FILE)
    else:
        preprocessed_data = joblib.load(PREPROCESSED_FILE)
    return preprocessed_data

def train_and_save_model(preprocessed_data):
    if not os.path.exists(MODEL_FILE):
        # Splitting the data, training the model
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_data['features'], preprocessed_data['label'], test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
    else:
        model = joblib.load(MODEL_FILE)
    return model

def main(operation):
    if operation == 'all' or operation == 'preprocess':
        preprocessed_data = load_and_preprocess_data()
        print("Data loading and preprocessing complete.")
    
    if operation == 'all' or operation == 'train':
        if 'preprocessed_data' not in locals():
            preprocessed_data = joblib.load(PREPROCESSED_FILE)
        model = train_and_save_model(preprocessed_data)
        print("Model training complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        operation = sys.argv[1]
        main(operation)
    else:
        print("Please specify an operation: 'all', 'preprocess', or 'train'")
