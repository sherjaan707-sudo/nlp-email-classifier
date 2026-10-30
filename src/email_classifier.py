# Email classifier
# Sorts customer emails into billing, technical and refund,
# and checks how similar two emails are.

from pathlib import Path

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Build the path from where this file lives, so the script finds the CSV
# whether I run it from PyCharm or from the terminal.
# The CSV itself is not in Git (see README).
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "emails_dataset.csv"


# 1. Load the dataset
df = pd.read_csv(DATA_PATH)
print("Number of emails:", len(df))


# 2. Clean the text with spaCy
# The small model is enough here. Cleaning only needs lemmas and
# stop-word flags, not word vectors.
nlp = spacy.load("en_core_web_sm")


# I wrapped cleaning in a function because I need it twice:
# once for the dataset and once for the new emails later.
def clean_text(text):
    doc = nlp(text)
    words = []
    for token in doc:
        # Stop words like "the" and "my" show up in every category,
        # so they don't help the model tell the classes apart.
        if not token.is_stop and not token.is_punct:
            # The lemma turns "charged" and "charges" into "charge",
            # so the model sees one word instead of several.
            words.append(token.lemma_.lower())
    return " ".join(words)


# Cleaning each email on its own learns nothing from the other emails,
# so it's safe to do this before the split.
df["clean_text"] = df["text"].apply(clean_text)

print("\nBefore cleaning:", df["text"][0])
print("After cleaning: ", df["clean_text"][0])


# 3. Split into training and test sets
# stratify keeps the same mix of labels in both sets. With only 30 emails,
# a random split could leave a category with almost no test examples.
# random_state=42 gives the same split every run, so results are repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print("\nTraining size:", len(X_train))
print("Test size:", len(X_test))


# 4. Turn text into TF-IDF features
# TF-IDF gives more weight to words like "invoice" or "crash" that are
# typical for some emails, and less to words that appear everywhere.
vectorizer = TfidfVectorizer()

# Fit on the training data only. If the vectoriser saw the test emails,
# test information would leak into training and accuracy would look too good.
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5. Train the model and check accuracy
# Logistic Regression is simple, fast and works well with TF-IDF.
# max_iter=1000 gives it enough steps to finish without a warning.
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# The test set is only 6 emails, so one mistake changes accuracy by about
# 17 percentage points. I treat this as a rough check, not a final verdict.
print("\nAccuracy:", round(accuracy, 2))


# 6. Try the model on new emails
# One email per category, written by me, to see how the model handles
# text it has never seen.
new_emails = [
    "I was billed twice this month.",
    "The app keeps crashing when I log in.",
    "I want my money back for the cancelled order.",
]

print("\nPredictions for new emails:")
for email in new_emails:
    # New emails must be cleaned the same way as the training emails,
    # otherwise the model sees words in a form it never learned.
    cleaned = clean_text(email)
    prediction = model.predict(vectorizer.transform([cleaned]))[0]
    print(prediction, "->", email)


# 7. Compare two emails from the dataset
# The medium model has word vectors, where words with similar meanings
# sit close together. The small model doesn't, so its scores aren't useful.
nlp_md = spacy.load("en_core_web_md")

# I compare the original text, because word vectors work on real words.
email_1 = df["text"][0]
email_2 = df["text"][1]

# similarity() averages the word vectors of each email, so it ignores
# word order. "refund, not a bill" and "bill, not a refund" score about the same.
similarity = nlp_md(email_1).similarity(nlp_md(email_2))

print("\nEmail 1:", email_1)
print("Email 2:", email_2)
print("Similarity:", round(similarity, 2))