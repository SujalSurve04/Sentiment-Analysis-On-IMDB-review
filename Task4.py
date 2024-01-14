import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample of the dataset
df = pd.read_csv('IMDB Dataset.csv').sample(n=1000, random_state=42)

# Explore data
print(df.info())
print(df.head())

# Preprocessing
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

df['preprocessed_text'] = df['review'].apply(preprocess_text)

# Training a Sentiment Analysis Model
model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
model.fit(df['preprocessed_text'], df['sentiment'])

# Predict sentiment
df['predicted_sentiment'] = model.predict(df['preprocessed_text'])

# Visualize sentiment distribution
sns.countplot(x='sentiment', data=df)
plt.title('Distribution of Actual Sentiments')
plt.show()

sns.countplot(x='predicted_sentiment', data=df)
plt.title('Distribution of Predicted Sentiments')
plt.show()
