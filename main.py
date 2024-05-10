import torch # Main PyTorch library for tensor operations and neural network modules
import transformers # Library for pre-trained models like BERT and utilities
from transformers import BertTokenizer # Tokeniser for converting text into tokens that BERT can understand
from transformers import BertForSequenceClassification # Pre-trained BERT model adapted for sequence classification tasks
from torch.utils.data import DataLoader, Dataset # Utilities to create and handle data batches during model training
import pandas as pd # Library for data manipulation and analysis, particularly useful for handling datasets
from sklearn.model_selection import train_test_split # Function to split the dataset into training and testing sets
from datasets import load_dataset # Function to easily load datasets
from tqdm import tqdm # Utility for displaying progress bars during operations like model training
import torch.nn as nn # PyTorch containing standard layers and activation functions
from torch.nn import CrossEntropyLoss # Loss function commonly used for classification tasks
from sklearn.metrics import f1_score, precision_score, recall_score # Metrics to evaluate the model performance
from A.BERT_BASE import BertAGNewsDataset, evaluate_BASEmodel # Import BertAGNewsDataset and evaluate_BASEmodel function

import tensorflow as tf  # TensorFlow is imported to leverage its deep learning capabilities.

from tensorflow.keras.preprocessing.sequence import pad_sequences  # Used for padding sequences to the same length
from tensorflow.keras.utils import to_categorical  # Converts class vectors to binary class matrices, useful for categorical cross-entropy
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenizer for text data, converting text to sequences of integers
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # For loading a tokenizer saved as a JSON file

from tensorflow.keras.models import load_model  # Import to load a saved Keras model
from tensorflow.keras.metrics import Precision, Recall  # Metrics to evaluate model performance, focusing on precision and recall

from tensorflow.keras import backend as K  # Importing Keras backend allows for more low-level operations, like tensor manipulations

import json  # Used for handling JSON files, often useful for reading or storing configurations and model architectures
import numpy as np  # NumPy is essential for handling numerical operations, especially arrays

#Configure to use mps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

#=====================BERT======================================================================================================================================================================

# Specify the pre-trained BERT model to use.
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4) #Define the number of labels in the dataset

# Load AG News dataset from Hugging Face datasets library
dataset = load_dataset('ag_news')

test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

test_dataset = BertAGNewsDataset(dataset['test'], tokenizer, max_length=256)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Evaluate the base model BERT
test_loss_Base, test_f1_Base, test_precision_Base, test_recall_Base = evaluate_BASEmodel(model, test_loader, device)

# Create a DataFrame to display results in a table
results_df_Base = pd.DataFrame({
    'Base Metric': ['Loss', 'F1 Score', 'Precision', 'Recall'],
    'Test Values': [f'{test_loss_Base:.4f}', f'{test_f1_Base:.4f}', f'{test_precision_Base:.4f}', f'{test_recall_Base:.4f}']
})


# Custom BERT class for classification tasks
class BERT(nn.Module):
    def __init__(self, num_classes=4):
        super(BERT, self).__init__()
        # Initialise a BERT model from the pretrained 'bert-base-uncased'.
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        # A linear layer to output the final classification result from the BERT model's output
        self.out = nn.Linear(768, num_classes)  # 768 dimensionality of BERT base's hidden layers

    def forward(self, ids, mask, token_type_ids):
        # Forward pass through BERT model. The model returns a tuple with various elements
        # where `o2` is the pooled output typically used for classification tasks
        _ , o2 = self.bert_model(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False  # Disable return of dictionary to ensure tuple output for backward compatibility
        )
        # Apply the linear layer on the pooled output `o2` to get the logits for each class
        out = self.out(o2)

        return out  # Return the logits from the linear layer
    
# Initialise an instance of the BERT class
model = BERT()

# Load a model from a saved .pth file, which is a PyTorch checkpoint. This line overrides 
# the previously initialised `model` with a model loaded from 'BERT_Fast.pth'.
model = torch.load('./A/BERT_Fast.pth')

# Transfer the model to the designated computation device, in this case mps. 
model = model.to(device)

#Define the Fine-Tune model of BERT
def evaluate_FineTune_model(model, dataloader, device):
    model.eval()  
    total_loss = 0
    all_preds = []  # Collect predictions from all batches to calculate classification metrics later
    all_labels = []  # Collect true labels for comparison
    loss_fn = CrossEntropyLoss()  # Set up the loss function, typical for classification tasks

    with torch.no_grad():  # Operations inside don't track gradients to reduce memory usage and speed up computation
        for dl in tqdm(dataloader, desc='Testing'):  # Process each batch and update the progress bar
            ids = dl['ids'].to(device)  # Load input IDs to the appropriate computation device
            token_type_ids = dl['token_type_ids'].to(device)  # Load segment IDs necessary for BERT-like models
            mask = dl['mask'].to(device)  # Load attention masks specifying where the model should focus
            labels = dl['target'].unsqueeze(1).to(device)  # Adjust labels for loss computation format

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)  # Generate predictions from the model
            loss = loss_fn(outputs, labels.view(-1).long())  # Calculate the loss between predictions and true labels
            total_loss += loss.item()  # Aggregate loss for reporting

            preds = outputs.argmax(dim=1).int()  # Determine the predicted classes from the model outputs
            all_preds.extend(preds.cpu().numpy())  # Transfer predictions to CPU and store it
            all_labels.extend(labels.view(-1).cpu().numpy())  # Same for labels

    #Calculate the metric values
    avg_loss = total_loss / len(dataloader)  
    f1 = f1_score(all_labels, all_preds, average='macro')  
    precision = precision_score(all_labels, all_preds, average='macro')  
    recall = recall_score(all_labels, all_preds, average='macro')  

    return avg_loss, f1, precision, recall  # Return values

# Evaluate the model
test_loss_Fine, test_f1_Fine, test_precision_Fine, test_recall_Fine = evaluate_FineTune_model(model, test_loader, device)

# Create a DataFrame to display results in a table
results_df_Fine = pd.DataFrame({
    'Fine-Tune Metric': ['Loss', 'F1 Score', 'Precision', 'Recall'],
    'Test Values': [f'{test_loss_Fine:.4f}', f'{test_f1_Fine:.4f}', f'{test_precision_Fine:.4f}', f'{test_recall_Fine:.4f}']
})



#=====================CNN-LSTM======================================================================================================================================================================

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)  # Initialises the parent class (Metric) with the name 'f1_score'
        self.precision = tf.keras.metrics.Precision()  # Creates a Precision object to calculate precision for predictions
        self.recall = tf.keras.metrics.Recall()  # Creates a Recall object to calculate recall for predictions

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)  # Update the precision calculation with the current batch
        self.recall.update_state(y_true, y_pred, sample_weight)  # Update the recall calculation with the current batch

    def result(self):
        p = self.precision.result()  # Retrieve the current precision value
        r = self.recall.result()  # Retrieve the current recall value
        # Return the calculated F1 score. K.epsilon() is used to prevent division by zero
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()  # Reset the internal state of the precision metric
        self.recall.reset_states()  # Reset the internal state of the recall metric



#======Without EDA

BASE_CNN_LSTM = load_model("./B/BASE_CNN-LSTM_classifier_model.h5", custom_objects={'F1Score': F1Score})  # Load the model  # Load the model  # Load the model

# Load tokenizer from JSON file
with open("./B/tokenizer_BASE.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    tokenizer_BASE = tokenizer_from_json(data)


test_sequences_BASE = tokenizer_BASE.texts_to_sequences(test_texts)
test_padded_sequences_BASE = pad_sequences(test_sequences_BASE, maxlen=180, padding='post')

test_labels = np.array(test_labels)
test_one_hot_labels = to_categorical(test_labels, num_classes=4)

# Evaluate the model
loss_CNNLSTM_BASE, accuracy_CNNLSTM_BASE, f1_CNNLSTM_BASE, precision_CNNLSTM_BASE, recall_CNNLSTM_BASE = BASE_CNN_LSTM.evaluate(test_padded_sequences_BASE, test_one_hot_labels)
results_df_CNNLSTM_BASE = pd.DataFrame({
    'BASE-CNN-LSTM Metric': ['Loss', 'F1 Score', 'Precision', 'Recall'],
    'Test Values': [f'{loss_CNNLSTM_BASE:.4f}', f'{f1_CNNLSTM_BASE:.4f}', f'{precision_CNNLSTM_BASE:.4f}', f'{recall_CNNLSTM_BASE:.4f}']
})


#=======EDA


EDA_CNN_LSTM = load_model("./B/EDA_CNN-LSTM_classifier_model.h5", custom_objects={'F1Score': F1Score})  # Load the model  # Load the model

# Load tokenizer from JSON file
with open("./B/tokenizer_EDA.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    tokenizer_EDA = tokenizer_from_json(data)


test_sequences_EDA = tokenizer_EDA.texts_to_sequences(test_texts)
test_padded_sequences_EDA = pad_sequences(test_sequences_EDA, maxlen=180, padding='post')

# Evaluate the model
loss_CNNLSTM_EDA, accuracy_CNNLSTM_EDA, f1_CNNLSTM_EDA, precision_CNNLSTM_EDA, recall_CNNLSTM_EDA = EDA_CNN_LSTM.evaluate(test_padded_sequences_EDA, test_one_hot_labels)

results_df_CNNLSTM_EDA = pd.DataFrame({
    'EDA-CNN-LSTM Metric': ['Loss', 'F1 Score', 'Precision', 'Recall'],
    'Test Values': [f'{loss_CNNLSTM_EDA:.4f}', f'{f1_CNNLSTM_EDA:.4f}', f'{precision_CNNLSTM_EDA:.4f}', f'{recall_CNNLSTM_EDA:.4f}']
})

print("========================================")
print("Results of BERT models")
print("========================================")
print(results_df_Base.to_string(index=False))
print(results_df_Fine.to_string(index=False))
print("========================================")
print("Results of CNN-LSTM models")
print("========================================")
print(results_df_CNNLSTM_BASE.to_string(index=False))
print(results_df_CNNLSTM_EDA.to_string(index=False))
print("========================================")