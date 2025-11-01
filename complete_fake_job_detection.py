# Fake Job Detection Baseline Pipeline


#Day 2 — Data Understanding

import pandas as pd

# Load dataset (after downloading from Kaggle)
df = pd.read_csv('fake_job_postings.csv')

# Display first few rows
print("Sample Data:")
print(df.head())


# Display basic info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Check distribution of target variable
print("\nTarget (fraudulent) Distribution:")
print(df['fraudulent'].value_counts())

# Basic statistics
print("\nDataset Summary:")
print(df.describe(include='all'))

# Insights:
print("""
Insights:
1 Many fake job postings have missing company profiles or descriptions.
2 Fake jobs often don't include company logos or have unrealistic salary ranges.
3 There is a strong class imbalance: real jobs >> fake jobs.
""")

#Day 3 — Text Cleaning and Preprocessing

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv('fake_job_postings.csv')

# Cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Apply cleaning
df['clean_description'] = df['description'].apply(clean_text)

# Compare before and after cleaning
print("Original:\n", df['description'].iloc[1][:300])
print("\nCleaned:\n", df['clean_description'].iloc[1][:300])

# Compare word count before vs after
df['word_count_before'] = df['description'].fillna('').apply(lambda x: len(str(x).split()))
df['word_count_after'] = df['clean_description'].apply(lambda x: len(x.split()))
print("\nAverage word count before:", df['word_count_before'].mean())
print("Average word count after:", df['word_count_after'].mean())

##  Day 3.5 — Feature Correlation and Text Insights

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset (make sure it includes 'clean_description')
df = pd.read_csv('fake_job_postings.csv')

# Select relevant columns
features = ['has_company_logo', 'telecommuting', 'employment_type', 'required_experience', 'fraudulent']
subset = df[features]

# Grouped analysis
print("\n--- Feature Distribution Grouped by Fraudulent ---")
for col in ['has_company_logo', 'telecommuting', 'employment_type']:
    print(f"\nFeature: {col}")
    print(df.groupby('fraudulent')[col].value_counts(normalize=True))

# Bar plots comparing proportions
plt.figure(figsize=(15, 5))

# 1 Company Logo presence
plt.subplot(1, 3, 1)
sns.countplot(x='has_company_logo', hue='fraudulent', data=df)
plt.title('Company Logo vs Fraudulent')
plt.xlabel('Has Company Logo (1 = Yes, 0 = No)')
plt.ylabel('Count')

# 2 Remote work availability
plt.subplot(1, 3, 2)
sns.countplot(x='telecommuting', hue='fraudulent', data=df)
plt.title('Remote Work (Telecommuting) vs Fraudulent')
plt.xlabel('Telecommuting (1 = Yes, 0 = No)')
plt.ylabel('Count')

# 3 Employment Type
plt.subplot(1, 3, 3)
sns.countplot(x='employment_type', hue='fraudulent', data=df)
plt.title('Employment Type vs Fraudulent')
plt.xlabel('Employment Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# WordClouds for real vs fake job posts
real_text = " ".join(df[df['fraudulent'] == 0]['description'].dropna())
fake_text = " ".join(df[df['fraudulent'] == 1]['description'].dropna())

real_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_text)
fake_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(fake_text)

# Display both word clouds
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.imshow(real_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Real Job Descriptions (Green)')

plt.subplot(1,2,2)
plt.imshow(fake_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Fake Job Descriptions (Red)')
plt.show()

##  Day 4 — Feature Extraction

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_description'])

print("TF-IDF Shape:", X_tfidf.shape)
print("Sample Features:", tfidf.get_feature_names_out()[:10])

# Top 15 words globally
sum_words = X_tfidf.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in tfidf.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

print("Top 15 Words with Highest TF-IDF Scores:")
for w, s in words_freq[:15]:
    print(f"{w}: {s:.4f}")


## Day 5 — Model Building and Evaluation


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare data
X = X_tfidf
y = df['fraudulent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("""
Interpretation:
Model achieves high accuracy because most jobs are real. However, recall for fake jobs may be lower.
Improving data balance or using class weights can help improve detection.
""")


## Day 6 — Model Analysis and Integration
import joblib
import numpy as np

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

# Pick 5 random examples
rand_idx = np.random.choice(len(y_test), 5, replace=False)
for i in rand_idx:
    text = df.iloc[y_test.index[i]]['description'][:200]
    print(f"\nJob Description (first 200 chars):\n{text}\nPredicted Probability of Fake: {probs[i]:.3f}")

# Save model and vectorizer
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\n Model and vectorizer saved successfully!")
