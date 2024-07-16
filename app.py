from flask import Flask, request, jsonify, render_template
import torch
import pickle
from transformers import BertTokenizerFast, AutoModel
import torch.nn as nn
import numpy as np

# Define BERT Architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the model and tokenizer
def load_model():
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load('c2_new_model_weights.pt', map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

app = Flask(__name__)
model, tokenizer = load_model()

# Prediction function
def predict(text, model, tokenizer):
    tokens = tokenizer.batch_encode_plus(
        [text],
        max_length=15,
        pad_to_max_length=True,
        truncation=True
    )
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])

    with torch.no_grad():
        preds = model(seq, mask)
        preds = preds.detach().cpu().numpy()
    
    prob = np.exp(preds[0][np.argmax(preds)])
    label = np.argmax(preds)
    label_str = "Fake" if label == 1 else "Real"
    return f"{label_str} news with probability {prob:.2f}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def classify_text():
    review = request.form['review']
    prediction = predict(review, model, tokenizer)
    return jsonify({'prediction_text': prediction})

if __name__ == '__main__':
    app.run(debug=True)
