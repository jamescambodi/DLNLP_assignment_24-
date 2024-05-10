# DLNLP_assignment_24-

# AG News Text Classification

## Project's Structure
- Folder A will have 3 files:
    - `BERT_BASE.py`: this file runs the base BERT on the AG News testing dataset directly
    - `Fine-Tune_BERT.ipynb`: this is the notebook contain all steps for running the fine-tuning process and saved as BERT_Fast.pth
    - `BERT_Fast.pth`: this preserve and retrieve a PyTorch model's state, in this case the fine-tuned BERT model

- Folder B will have 6 files:
    - `CNN-LSTM_Trainer.ipynb`: This Jupyter notebook contains the core training scripts for a CNN-LSTM hybrid model. It includes detailed steps for setting up the model, training it on your dataset, and evaluating its performance.
    - `EDA_CNN-LSTM_Trainer.ipynb`: This Jupyter notebook extends `CNN-LSTM_Trainer.ipynb` by incorporating Exploratory Data Analysis (EDA). It provides insights into the data characteristics and distributions that are crucial for refining model training strategies. 
    - `BASE_CNN-LSTM_classifier_model.h5`: This file is a saved H5 model file containing the base CNN-LSTM classifier. This model can be directly loaded and used for predictions without the need for retraining, serving as a quick-start for deployment or further fine-tuning.
    - `EDA_CNN-LSTM_classifier_model.h5`: Similar to the base model, this H5 file contains a CNN-LSTM classifier that has been optimised and fine-tuned after comprehensive exploratory data analysis. 
    - `tokenizer_BASE.json`: This JSON file contains the serialised form of the tokenizer used with the BASE CNN-LSTM model and saved its trained tokenizer. The tokenizer is essential for converting text data into sequences that the neural network can process. 
    - `tokenizer_EDA.json`: This JSON file holds the tokenizer used with the EDA-optimized CNN-LSTM model and saved its trained tokenizer. It includes adjustments derived from exploratory data analysis, ensuring that the text preprocessing aligns with insights gained from the EDA. 

- `main.py` this is the python script include all the necessary steps to run the Text Classification of AG News end-to-end and report the performance metric of each model.

## Packages Required

These are libraries required before running the code. 

### PyTorch and TensorFlow
- `torch`: The primary library for tensor operations and neural network modules. 
- `torch.nn`: This submodule contains standard neural network layers and activation functions necessary for constructing our models.
- `tensorflow`: Supports deep learning tasks, especially in data preprocessing and evaluation 

### Hugging Face Transformers
- `transformers`: This library provides access to pre-trained models like BERT, along with utilities for effectively using these models in natural language processing (NLP) tasks.
- `BertTokenizer` and `BertForSequenceClassification`: Specific components from the transformers library used for tokenizing text data into a format understandable by BERT models and adapting BERT for sequence classification tasks, respectively.

### Data Handling and Processing
- `pandas`: for reading dataframe.
- `DataLoader` and `Dataset` from `torch.utils.data`: These tools help in creating and managing data batches during the training process, ensuring efficient data handling.

### Machine Learning Utilities
- `train_test_split` from `sklearn.model_selection`: A utility for splitting the dataset into training and testing portions, vital for validating the model's performance.
- `f1_score`, `precision_score`, `recall_score` from `sklearn.metrics`: Metrics used to evaluate the accuracy and effectiveness of the model on test data.

### Additional TensorFlow Utilities
- **Preprocessing and Model Evaluation**: Prepares text data and evaluates models using precision and recall metrics (`tensorflow.keras.preprocessing`, `tensorflow.keras.metrics`).

### Miscellaneous
- `tqdm`: For showing the progress bar.
- `numpy`: for handling the vector/matrix operations (add, substract, divide, etc).
- `json`: Necessary for managing JSON files, typically used for storing model configurations or architecture details.

## How to run
### 1. Download data
- You have to download the AG News Dataset

### 2. Inference
```python
$ python main.py
```