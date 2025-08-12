# ✈️ Twitter US Airline Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-green?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on **classifying customer sentiments** about US airlines on Twitter into three categories:
- **Positive** 😀
- **Negative** 😠
- **Neutral** 😐

It uses **NLP preprocessing techniques** and **Machine Learning models** to analyze tweets and predict sentiment.  
The workflow includes:
- Data Cleaning
- Tokenization & Stopword Removal
- Abbreviation Expansion
- TF-IDF Feature Extraction
- Logistic Regression & Support Vector Machine (SVM) Models

---

## 📂 Dataset
**Source**: Twitter US Airline Sentiment dataset  
- **Training Data**: 10,980 tweets (12 columns)
- **Test Data**: Unlabeled tweets for prediction

Example:
| airline_sentiment | airline   | text                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| negative          | Southwest | scheduled morning, 2 days fact, yes..not sure...                    |
| positive          | United    | flew ord miami back great crew, service legs...                     |
| negative          | United    | ord delayed air force one, last sbn 8:20, 5 minutes delay...         |

---

## 🛠 Technologies & Libraries
- **Python** (Pandas, NumPy, Regex, NLTK)
- **NLP**: Tokenization, Stopwords, Abbreviation Handling, Emoji Removal
- **Feature Engineering**: `TfidfVectorizer`
- **Machine Learning Models**:
  - Logistic Regression (`scikit-learn`)
  - Support Vector Machine (SVC with Linear Kernel)
- **Evaluation**: Predictions saved for submission

---

## ⚙️ Data Preprocessing Steps

1️⃣ **Drop Unnecessary Columns**  
Removed columns like `name`, `tweet_id`, `retweet_count`, `user_timezone` to focus on text data.

2️⃣ **Stopword Removal & Punctuation Removal**  
- Used `nltk.corpus.stopwords`
- Added airline-related words (`flight`, `airline`, `flights`, `AA`) to stopwords.

3️⃣ **Abbreviation Expansion**  
Converted common abbreviations like:
ppl → people
cust → customer
serv → service
mins → minutes

4️⃣ **Text Cleaning**  
- Removed URLs, usernames (`@user`), hashtags (`#word` → `word`), extra whitespaces
- Removed emojis and numeric tokens

5️⃣ **Final Text Format**  
Added airline name and negative reason (if available) to the tweet for additional context.

---

## 🔍 Feature Extraction
Used **TF-IDF Vectorization**:
```python
v = TfidfVectorizer(analyzer='word', max_features=3150, max_df=0.8, ngram_range=(1,1))
train_features = v.fit_transform(train.text)
test_features = v.transform(test.text)


🤖 Models & Training
1️⃣ Logistic Regression
clf = LogisticRegression(C=2.1, solver='liblinear', multi_class='auto')
clf.fit(train_features, train['airline_sentiment'])
pred = clf.predict(test_features)
Regularization parameter C = 2.1

Saved predictions to predictions_twitter.csv

2️⃣ Support Vector Machine (SVM)
python
Copy
Edit
clf = SVC(kernel="linear", C=0.96, gamma='scale')
clf.fit(train_features, train['airline_sentiment'])
pred = clf.predict(test_features)
Linear kernel, C = 0.96

Saved predictions to predictions_twitter2.csv

📊 Results & Insights
Logistic Regression performed better than SVM in this setup.

Tweets with clear sentiment words (like great, love, delay, cancelled) were classified with high confidence.

Airline names and negative reason tags improved classification accuracy.

🚀 How to Run
1️⃣ Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/twitter-airline-sentiment.git
cd twitter-airline-sentiment
2️⃣ Install dependencies:
bash
Copy
Edit
pip install pandas numpy scikit-learn nltk
3️⃣ Download NLTK stopwords:
python
Copy
Edit
import nltk
nltk.download('stopwords')
4️⃣ Place datasets in the correct directory and run:
bash
Copy
Edit
python sentiment_analysis.py
📌 Author
SK Samim Ali
📧 roy871858@gmail.com
