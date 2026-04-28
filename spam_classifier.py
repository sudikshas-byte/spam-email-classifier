import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Select required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom input
while True:
    user_input = input("\nEnter a message (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break

    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)

    if prediction[0] == 1:
        print("Spam Message 🚫")
    else:
        print("Not Spam ✅")
