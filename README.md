# SPAM MAIL DETECTION 
This project builds a Spam Email Classifier using TF-IDF vectorization and Logistic Regression.

📌 Project Summary

✓ Loads dataset
✓ Cleans & preprocesses email text
✓ Converts text into numerical features using TF-IDF
✓ Splits data into Train/Test
✓ Trains Logistic Regression model
✓ Evaluates accuracy
✓ Predicts whether new mail is spam or ham

📁 Dataset Used

The dataset contains:

✓ Category → spam / ham
✓ Message → email text
✓ Sometimes an unwanted third column (\tCategory)

Dataset shape:

(5572, 3)


Example rows:

Category	Message
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup...
🧹 Data Preprocessing

✓ Handled missing values
✓ Converted labels: spam → 0, ham → 1
✓ Split into features (X) and labels (Y)

✂️ Train-Test Split

✓ 80% → Training
✓ 20% → Testing
✓ Random state = 3

Rows:

✓ Training → 4457
✓ Testing → 1115

🔠 TF-IDF Feature Extraction

Using:

TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)


✓ Converts text into TF-IDF numerical matrix
✓ Removes common English stopwords

🤖 Model Used: Logistic Regression
model = LogisticRegression()
model.fit(X_train_features, Y_train)


✓ Simple
✓ Fast
✓ Works well for text classification

📊 Model Accuracy

✓ Training Accuracy: 96.77%
✓ Testing Accuracy: 96.68%

The model performs consistently and is not overfitting.

💡 Predictive System

Example:

input_mail = ["I've been searching for the right words to thank you ..."]


Prediction →
✓ ham mail

▶️ How to Run

✓ Upload dataset in Google Colab
✓ Paste the code
✓ Run all cells
✓ Enter any message to classify as spam/ham

📂 Technologies Used

✓ Python
✓ Pandas
✓ NumPy
✓ Scikit-learn
✓ TF-IDF Vectorizer
✓ Logistic Regression
✓ Google Colab

📈 Future Improvements

✓ Add Naive Bayes model comparison
✓ Clean text (links, numbers, punctuation, HTML)
✓ Deploy with Flask/Streamlit
✓ Build a UI for predictions
