from flask import Flask, render_template, request
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences 

app = Flask(__name__)

# Load the code dataset and model
data = pd.read_csv('code.csv')

# Encode the expected_output column
label_encoder = LabelEncoder()
data['expected_output_encoded'] = label_encoder.fit_transform(data['expected_output'])

# Tokenize the code snippets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['code_snippet'])
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences and pad them to a fixed length
input_sequences = tokenizer.texts_to_sequences(data['code_snippet'])
padded_sequences = pad_sequences(input_sequences)

# Define the maximum sequence length used during training
MAX_SEQUENCE_LENGTH = 100

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

# Instantiate the model with the same parameters
embedding_dim = 50
hidden_dim = 100
output_dim = len(data['expected_output'].unique())  # Number of unique error types
model = LSTMModel(embedding_dim, hidden_dim, total_words, output_dim)

# Load the saved model
model.load_state_dict(torch.load('lstm_model.pth'))

# Function to predict error type for a given code snippet
def predict_error(code_snippet, tokenizer):
    # Tokenize and pad the code snippet
    input_sequence = tokenizer.texts_to_sequences([code_snippet])
    padded_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    input_tensor = torch.tensor(padded_sequence, dtype=torch.long)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_label = torch.max(output, 1)

    # Convert the predicted label to the original error type
    predicted_error = label_encoder.classes_[predicted_label.item()]
    
    return predicted_error

# Function to get error output using existing tool if model fails to predict
def get_error_output(code):
    global_vars = {}
    error_report = []

    for line_number, line in enumerate(code.split('\n'), 1):
        try:
            exec(line, global_vars)
        except Exception as e:
            error_message = f"Error on line {line_number}: {str(e)}"
            error_report.append(error_message)
    
    return error_report

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        code_to_predict = request.form['code_snippet']

        # Predict error type using the model
        predicted_error = predict_error(code_to_predict, tokenizer)

        # If the model fails to predict, use the existing tool
        if predicted_error is None:
            error_output = get_error_output(code_to_predict)
            return render_template('index.html', code=code_to_predict, error_output=error_output)
        
        # Display the predicted error in the UI
        return render_template('index.html', code=code_to_predict, predicted_error=predicted_error)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
