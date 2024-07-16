# Fake News Detection using Bert
<br>
<b>Project Overview</b>
This project focuses on detecting fake news using the BERT (Bidirectional Encoder Representations from Transformers) model. The primary goal is to build a robust machine learning model that can accurately classify news articles as either true or fake. The project involves several stages, including data preprocessing, model training, and evaluation.
<br>

# Introduction
The proliferation of fake news has become a significant issue in today's digital age. Fake news can mislead the public, influence political decisions, and create social unrest. This project aims to address this problem by leveraging the BERT model to detect fake news with high accuracy.
<br>

# Dataset Description
<br>
The dataset used in this project consists of two CSV files: one containing true news articles and the other containing fake news articles. Each article includes the following attributes:
<br>

The dataset is availabe on [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)<br>

<b>Title: </b>The headline of the news article.<br>
<b>Text: </b>The main content of the news article.<br>
<b>Subject:</b> The category or subject of the news article.<br>
<b>Date: </b>The publication date of the news article.<br>
<b>Target: </b>A label indicating whether the news is true or fake.<br>

The true and fake news articles are merged into a single DataFrame, and a new column, label, is created to represent the target labels numerically (0 for true and 1 for fake).
<br>

# Plan Of Action
<br>

![alt text](images/plan_of_action.png)
<br>

# Data Preprocessing
<br>
Data preprocessing is a crucial step in preparing the dataset for model training. The following steps were performed:
<br>
<b>Label Encoding:</b> The target labels (True/Fake) were converted to numerical values (0/1).<br>
<b>Data Balancing: </b>The dataset was checked for balance between true and fake news articles.<br>
<b>Train-Test Split:</b> The dataset was split into training, validation, and test sets in a 70:15:15 ratio.<br>
<b>Tokenization:</b> The BERT tokenizer was used to tokenize and encode the text data into a format suitable for the BERT model.<br>

# Model Architecture
<br>
The BERT model was fine-tuned for the task of fake news detection. The architecture includes:
<br>
<b>BERT Base Model: </b>Pre-trained BERT model (bert-base-uncased) from HuggingFace.<br>
<b>Dropout Layer: </b>To prevent overfitting.<br>
<b>ReLU Activation: </b>For non-linearity.<br>
<b>Fully Connected Layers:</b> Two dense layers for classification.<br>
<b>Softmax Activation: </b>For outputting probabilities of the classes.<br>

![alt text](images/Fine_tuning.png)

# Training and Evaluation<br>
The model was trained using the AdamW optimizer and a negative log-likelihood loss function. The training process involved:
<br>
<b>Freezing BERT Layers: </b>Only the final layers were fine-tuned to speed up training and prevent overfitting.<br>
<b>Training Loop: </b>The model was trained for a specified number of epochs, with periodic evaluation on the validation set.<br>
<b>Evaluation Metrics: </b>The model's performance was evaluated using precision, recall, f1-score, and accuracy.<br>

# Results<br>
The model achieved an accuracy of 86% on the test set. The detailed classification report is as follows:<br>

![alt text](images/classification_report.png)
<br>
The model demonstrated a balanced performance across both classes, indicating its effectiveness in detecting fake news.<br>

# Web Application
<br>

![alt text](images/web_app.png)
<br>

# Conclusion
<br>
This project successfully developed a BERT-based model for fake news detection with high accuracy. The model can be further improved by:
<br>
Increasing Training Data: More diverse and larger datasets can enhance model performance.<br>
Hyperparameter Tuning: Fine-tuning hyperparameters for optimal performance.<br>
Advanced Architectures: Exploring more advanced transformer architectures or ensemble methods.<br>