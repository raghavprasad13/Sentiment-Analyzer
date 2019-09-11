# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from googletrans import Translator
# Importing the dataset
#we make use of a tab sep value as commas are presnt in our text!
#read_csv expects csv as object, but tsv hai toh make use of arguments
#quoting=3 ignores " "
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts-- getting only the relevant words, punctuation,numbers(that don't have any significance)
#stemming is used to get rid of loved, loves, love as a same word to prevent too many words in our bag of words
#getting rid of Capital letters- only lowercase 
import re
import nltk
nltk.download('stopwords') #stopwords contain all the irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#corpus is a collection of text
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #" " replaces the chars other than alphabets
    review = review.lower()
    #review is a strig, split converts the string into words by using space as seperator
    review = review.split()
    ps = PorterStemmer()
    #set used for faster execution than list
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #to convert back review from list to string use join
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
cv=Translator.translate(cv)
print(cv)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)