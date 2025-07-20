# IMDB-Movie-Review-Sentiment-Analysis

🎬 IMDB Movie Review Sentiment Analysis
This project performs sentiment analysis on IMDB movie reviews using a Simple Recurrent Neural Network (RNN). The model classifies movie reviews as positive or negative based on the review text.

📌 Project Overview
✅ Dataset: IMDB movie reviews (pre-tokenized)

✅ Task: Sentiment classification (positive/negative)

✅ Model: Simple RNN

✅ Tools: Python, TensorFlow, Keras, NumPy

🧠 Model Architecture
Embedding Layer – Converts word indices into dense vectors

SimpleRNN Layer – Captures temporal dependencies in word sequences

Dense Layer – Final layer with sigmoid activation for binary classification

🔧 Project Setup
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/IMDB-Movie-Review-Sentiment-Analysis.git
cd IMDB-Movie-Review-Sentiment-Analysis
2. Install Dependencies
Make sure you have Python ≥ 3.7. Then run:

bash
Copy
Edit
pip install -r requirements.txt
3. Run the Notebook / Script
You can run the Jupyter notebook or Python script to train and test the model.

🚀 Features
Load and preprocess IMDB dataset

Decode integer-encoded reviews into readable text

Pad sequences for consistent input size

Train with EarlyStopping to prevent overfitting

Predict custom user review sentiment

🧪 Sample Usage
python
Copy
Edit
# Preprocess a new review
review = "This movie was boring and too long"
processed = preprocess_text(review)

# Make prediction
prediction = model.predict(processed)
print("Sentiment:", "Positive" if prediction[0][0] > 0.5 else "Negative")
📊 Model Evaluation
Trained on 25,000 IMDB reviews

Evaluated on a separate 25,000 review test set

Metrics: Accuracy, loss, and custom text predictions

🧾 Dataset Info
The IMDB dataset includes:

25,000 training samples

25,000 testing samples

Binary sentiment labels: 0 = Negative, 1 = Positive

Reviews are already preprocessed into integer sequences

📎 Folder Structure
text
Copy
Edit
IMDB-Movie-Review-Sentiment-Analysis/
│
├── imdb_sentiment_analysis.ipynb     # Main notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── model.h5                          # Saved model weights (optional)
📚 References
Keras IMDB Dataset Documentation

TensorFlow RNN Guide

[Deep Learning with Python - François Chollet]

🙌 Acknowledgements
Thanks to the creators of Keras and TensorFlow for providing powerful tools for deep learning development!

📬 Contact
Author: Niyanta

Email: niyanta.official@gmail.com

LinkedIn: linkedin.com/in/niyanta02/
