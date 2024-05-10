import torch # Main PyTorch library for tensor operations and neural network modules
import transformers # Library for pre-trained models like BERT and utilities
from transformers import BertTokenizer # Tokeniser for converting text into tokens that BERT can understand
from transformers import BertForSequenceClassification # Pre-trained BERT model adapted for sequence classification tasks
from torch.utils.data import DataLoader, Dataset # Utilities to create and handle data batches during model training
import pandas as pd # Library for data manipulation and analysis, particularly useful for handling datasets
from sklearn.model_selection import train_test_split # Function to split the dataset into training and testing sets
from datasets import load_dataset # Function to easily load datasets, especially useful for standard NLP tasks
from tqdm import tqdm # Utility for displaying progress bars during operations like model training
import torch.nn as nn # PyTorch containing standard layers and activation functions
from torch.nn import CrossEntropyLoss # Loss function commonly used for classification tasks
from sklearn.metrics import f1_score, precision_score, recall_score # Metrics to evaluate the model performance

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define a PyTorch Dataset for the AG News dataset that processes text through a tokeniser for BERT
class BertAGNewsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        # Initialise the dataset with a list or dataset object containing news data, a tokeniser, and a maximum token length
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch a single news article by index and process it for BERT consumption
        text = self.dataset[idx]["text"]  # Extract text for the given index
        label = self.dataset[idx]["label"]  # Extract label for the given index

        # Tokenise the text using BERT's tokeniser. The tokeniser converts text into token ids,
        # generates attention masks, and token type ids
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,  # Ensure you add this
            padding='max_length',  # Pad or truncate the sequence to a uniform length
            max_length=self.max_length, # Max length to which the text is padded or truncated
            add_special_tokens=True,  # Add tokens like [CLS], [SEP] for BERT to work properly
            return_attention_mask=True,  # Include attention masks in the returned dictionary
        )

        # Prepare the output dictionary which aligns with what the model expects in terms of input tensors
        return {
            'ids': torch.tensor(inputs["input_ids"], dtype=torch.long),  # Input IDs for BERT model
            'mask': torch.tensor(inputs["attention_mask"], dtype=torch.long),  # Attention masks
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),  # Token type IDs
            'target': torch.tensor(label, dtype=torch.long)  # Labels for each text item
        }

def evaluate_BASEmodel(model, dataloader, device):
    model = model.to(device)  # Move the model to the device
    model.eval()  # Set the model to evaluation mode
    total_loss = 0  # Initialise total loss to accumulate loss values
    all_preds = []  # List to store all predictions
    all_labels = []  # List to store all true labels
    
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function, suitable for multi-class classification tasks

    with torch.no_grad():  # Disable gradient calculation to speed up the process and reduce memory usage
        for batch in tqdm(dataloader, desc='Testing'):  # Iterate over each batch in the dataloader
            input_ids = batch['ids'].to(device)  # Move input ids to the device
            attention_mask = batch['mask'].to(device)  # Move attention masks to the device
            labels = batch['target'].to(device)  # Move labels to the device

            # Pass the inputs through the model. The model outputs logits.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Calculate loss for the current batch by comparing the model output and the true labels
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()  # Accumulate the batch loss

            # Calculate the predictions by selecting the maximum logit value from the output logits
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Calculate F1 score, precision, and recall
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Return the average loss, F1 score, precision, and recall
    return avg_loss, f1, precision, recall

if __name__ == '__main__':
    print("Using device:", device)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    dataset = load_dataset('ag_news')
    test_dataset = BertAGNewsDataset(dataset['test'], tokenizer, max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    test_loss_Base, test_f1_Base, test_precision_Base, test_recall_Base = evaluate_BASEmodel(model, test_loader, device)
    results_df_Base = pd.DataFrame({
        'Base Metric': ['Loss', 'F1 Score', 'Precision', 'Recall'],
        'Test Values': [f'{test_loss_Base:.4f}', f'{test_f1_Base:.4f}', f'{test_precision_Base:.4f}', f'{test_recall_Base:.4f}']
    })
    
    print(results_df_Base.to_string(index=False))