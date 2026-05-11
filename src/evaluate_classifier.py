"""
--Part 4: Evaluation--
This script loads the trained neural network model from Part 3 along
with the sentence embeddings and topic labels, then evaluates model
performance by generating predictions, calculating accuracy, and
displaying a confusion matrix.
"""
import argparse # pass in filepaths from terminal
import pandas as pd # manage table data
import torch # PyTorch library
import torch.nn as nn # neural network (NN) tools
from sklearn.preprocessing import LabelEncoder # convert text labels into numeric class labels
from sklearn.metrics import accuracy_score, confusion_matrix # evaluation metrics

# recreate the same neural network architecture from Part 3 in order to load saved model weights
class TopicClassifier(nn.Module):

    # initialize NN
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super().__init__() # initialize parent NN class
        self.linear1 = nn.Linear(input_dim, hidden_dim) # first NN layer: transfer embeddings to hidden layer
        self.relu = nn.ReLU() # activation function for hidden layer
        self.linear2 = nn.Linear(hidden_dim, output_dim) # output layer: output topic scores

    # how NN processes input data
    def forward(self, x):
        x = self.linear1(x) # pass input embeddings through first layer (linear)
        x = self.relu(x) # apply ReLU activation function (non-linear)
        x = self.linear2(x) # pass through output layer (linear); output topic scores
        return x # return raw class scores

# pipeline for Part 4 evaluation 
def main(args):# args contains parsed command-line arguments

    # initialization status checks
    print("\nEvaluating classifier...") # print status label
    print(f"\nTraining TSV: {args.train_tsv}") # print path to training file
    print(f"Embeddings file: {args.embeddings}") # print path to embeddings (part 2)
    print(f"Model file: {args.model}") # print path to model

    # load labels and embedding tables from TSV files via pandas
    labels_df = pd.read_csv(args.train_tsv, sep="\t") # load original training TSV containing labels and text
    embeddings_df = pd.read_csv(args.embeddings, sep="\t") # load sentence embeddings from part 2

    # convert text labels into numeric classes
    label_encoder = LabelEncoder() # create label encoder object
    y_true = label_encoder.fit_transform(labels_df["category"]) # convert text topic labels into numeric class labels for evaluation
    X = embeddings_df.values # convert embeddings into numeric feature matrix
    X_tensor = torch.tensor(X, dtype=torch.float32) # convert feature matrix to PyTorh tensor format (float)

    # data inspection status check
    print("\nData loaded for evaluation.") # print status label
    print(f"\nX shape (rows, dimensions): {X_tensor.shape}") # print shape of feature matrix (rows, dimensions)
    print(f"y shape (examples): {y_true.shape}") # print shape of label array (number of examples)

    input_dim = X_tensor.shape[1] # number of vector dimensions
    hidden_dim = 128 # size of middle layer (evaluatoin 1 = 64, evaluation 2 = 128)
    output_dim = len(label_encoder.classes_) # number of topic categories

    model = TopicClassifier(input_dim, hidden_dim, output_dim) # create NN object (same as training model))
    model.load_state_dict(torch.load(args.model)) # load saved weights from trained model
    model.eval() # switch to evaluative mode

    print("\nModel loaded for evaluation.") # print status label

    with torch.no_grad(): # gradient calculations not needed for evaluation
        predictions = model(X_tensor) # generate topic prediction scores

    predicted_labels = torch.argmax(predictions, dim=1).numpy() # select highest-scoring class prediction

    accuracy = accuracy_score(y_true, predicted_labels) # calculate accuracy
    conf_matrix = confusion_matrix(y_true, predicted_labels) # generate confusion matrix

    # print evaluation
    short_labels = [
    "ent",
    "geo",
    "health",
    "pol",
    "sci/tech",
    "sports",
    "travel"
    ]

    print(f"\nAccuracy: {accuracy:.4f}") # print classification accuracy
    print("\nConfusion Matrix:") # print label
    #print("\nLabel classes:") # print label
    #print(label_encoder.classes_) # print classes

    confusion_df = pd.DataFrame( # improved confusion matrix
        conf_matrix,
        index=label_encoder.classes_,
        columns=short_labels # label_encoder.classes_
    )    

    print(confusion_df.to_string(col_space=10, justify="center")) # print confusion matrix
    print("\nRows = true labels") # print row labels
    print("Columns = predicted labels") # print column label
    print() # print space for readability


# define command-line arguments    
if __name__ == "__main__": # run only if this file is executed directly

    parser = argparse.ArgumentParser() # create parser for command-line arguments
    parser.add_argument("--train_tsv", type=str, required=True) # add required command-line path argument for training TSV
    parser.add_argument("--embeddings", type=str, required=True) # add required command-line path argument for embeddings file
    parser.add_argument("--model", type=str, required=True) # add required command-line path argument for saved model file
    args = parser.parse_args() # parse command-line input into usable Python variables

    main(args) # use command-line arguments to initiate evaluation pipeline