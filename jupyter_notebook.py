#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the JSON dataset
data_path = r"D:\\Work\\projects\\news_recommendation_app\\News_Category_Dataset_v3.json"

# Load JSON data directly into a DataFrame
df = pd.read_json(data_path, lines=True)
df.head()


# In[2]:


#initial data exploration
# Display the first few rows of the DataFrame
df.head()

# Check for missing values
df.isnull().sum()

# Basic statistics
df.describe()

# Data types
df.dtypes


# In[2]:


#handle missing values
# Drop rows with missing values
df = df.dropna()

# Drop duplicate rows
df = df.drop_duplicates()

# Verify the cleaning steps
df.isnull().sum()
df.duplicated().sum()


# In[3]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'description' column
df['processed_content'] = df['short_description'].apply(preprocess_text)

# Display the first few rows of the DataFrame after preprocessing
print(df[['short_description', 'processed_content']].head())


# In[5]:


df['content_length'] = df['processed_content'].apply(len)


# In[8]:


print(df['content_length'].isnull().sum())  # Check for NaN values


# In[9]:


print(df['content_length'])

print(df['processed_content'])


# In[6]:


import matplotlib.pyplot as plt

# Drop NaN values in 'content_length' column if any
df_clean = df.dropna(subset=['content_length'])

plt.figure(figsize=(10, 6))
plt.hist(df_clean['content_length'], bins=50, density=True, alpha=0.6, color='g')
plt.title('Distribution of Article Lengths')
plt.xlabel('Content Length')
plt.ylabel('Frequency')
plt.show()


# In[7]:


from wordcloud import WordCloud

# Combine all processed content into a single string
text = ' '.join(df['processed_content'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Article Content')
plt.show()


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the processed content
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_content'])


# In[29]:


from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np

# Perform LSA
n_components = 100  # Number of latent topics
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

tfidf_matrix_reduced = lsa.fit_transform(tfidf_matrix)


# In[30]:


# Function to get similar articles
def get_similar_articles(input_text, num_recommendations=5):
    input_vector = tfidf_vectorizer.transform([input_text])
    input_vector_reduced = lsa.transform(input_vector)
    cosine_similarities = np.dot(tfidf_matrix_reduced, input_vector_reduced.T).flatten()
    similar_indices = cosine_similarities.argsort()[::-1][:num_recommendations]
    return df.iloc[similar_indices]


# In[ ]:


# Interactive session to get recommendations
def recommend_articles():
    while True:
        try:
            input_text = input("Enter a few keywords or a phrase to get news recommendations (or type 'exit' to quit): ")
            if input_text.lower() == 'exit':
                break
            
            recommendations = get_similar_articles(input_text)
            print(f"Recommendations for your input '{input_text}':")
            for idx, row in recommendations.iterrows():
                print(f"Article ID: {idx}, Headline: {row['headline']}, Short Description: {row['short_description']}")
        except Exception as e:
            print(f"An error occurred: {e}")

# Start the recommendation session
recommend_articles()


# In[ ]:




