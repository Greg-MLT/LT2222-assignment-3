"""
--Part 3: Neural Topic Classification--
This script loads sentence embeddings from Part 2 and their
corresponding topic labels, converts the data into PyTorch tensors,
and trains a feed-forward neural network classifier for multiclass
topic prediction. The trained model is saved as a .pt file.
"""
import argparse # pass in filepaths from terminal
import pandas as pd # manage table data
import numpy as np # math: convert data into arrays/numeric transformations
from sklearn.preprocessing import LabelEncoder # convert topic labels to machine-readable numeric classes
import torch # PyTorch library
import torch.nn as nn # neural network (NN) tools
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt # plotting tools

# define class: neural classifier
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

# pipline for Part 3 neural topic classification
def main(args): # args contains parsed command-line arguments

    # initialization status checks
    print("Training classifier...") # classifier script started
    print(f"Training TSV: {args.train_tsv}") # print path to training file
    print(f"Embeddings file: {args.embeddings}") # print path to embeddings (part 2)
    print(f"Output model: {args.output_model}") # print location of trained model

    # load labels and embedding tables from TSV files via pandas
    labels_df = pd.read_csv(args.train_tsv, sep="\t") # load original training TSV containing labels and text
    embeddings_df = pd.read_csv(args.embeddings, sep="\t") # load sentence embeddings from part 2

    # data inspection status checks
    print(f"Labels shape: {labels_df.shape}") # print size of labels table (rows, columns)
    print(f"Embeddings shape: {embeddings_df.shape}") # print size of embeddings table (rows, dimensions)
    print("First labels:") # print status label
    print(labels_df["category"].head()) # print first 5 labels

    # convert text labels into numeric classes
    label_encoder = LabelEncoder() # create label encoder object
    y = label_encoder.fit_transform(labels_df["category"]) # learn label mappings and convert labels into integers
    print("Encoded labels:") # print status label
    print(y[:5]) # print first 5 encoded labels (y)

    X = embeddings_df.values # assign sentence embeddings to feature matrix x

    print("Feature matrix shape:") # print status label
    print(X.shape) # print shape of feature matrix (rows, dimensions)
    print("First feature vector:") # print status label
    print(X[0][:10]) # print first 10 dimensions of first embedding vector

    # convert training data into PyTorch tensor format
    X_tensor = torch.tensor(X, dtype=torch.float32) # convert embeddings feature matrix into tensor format (float format)
    y_tensor = torch.tensor(y, dtype=torch.long) # convert encoded labels into tensor format (long int format)

    # added for bonus 1: validation
    if args.dev_tsv and args.dev_embeddings: # bonus 1: load validation data if both validation arguments are provided
        dev_labels_df = pd.read_csv(args.dev_tsv, sep="\t") # bonus 1: load validation labels into pandas tables
        dev_embeddings_df = pd.read_csv(args.dev_embeddings, sep="\t") # bonus 1: load validation embeddings into pandas tables

        y_dev = label_encoder.transform(dev_labels_df["category"]) # bonus 1: convert validation text labels into class IDs
        X_dev = dev_embeddings_df.values # bonus 1: convert validation embeddings table into feature matrix

        X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32) # bonus 1: convert to PyTorch tensor format
        y_dev_tensor = torch.tensor(y_dev, dtype=torch.long) # bonus 1: convert to PyTorch tensor format

        # bonus 1: validation data staus check
        print("Validation data loaded.") # bonus 1: print status label
        print(f"Validation X shape: {X_dev_tensor.shape}") # bonus 1: print x shape
        print(f"Validation y shape: {y_dev_tensor.shape}") # bonus 1: print y shape

    else: # bonus 1: fallback if no validation data
        X_dev_tensor = None # bonus 1: no validation active
        y_dev_tensor = None # bonus 1:  no validation active
        print("No validation data provided.") # bonus 1: print status label

    print("Tensor shapes:") # print status label
    print(X_tensor.shape) # print x tensor sizes (examples, dimensions)
    print(y_tensor.shape) # print y tensor shape (number of training examples)

    input_dim = X_tensor.shape[1] # number of vector dimensions
    hidden_dim = 128 # size of middle layer (initial evaluation = 64, 2nd evaluation = 128)
    output_dim = len(label_encoder.classes_) # number of topic categories

    model = TopicClassifier(input_dim, hidden_dim, output_dim) # create the NN object using chosen layer sizes

    # note: added class weighting to loss function later training runs
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts 
    class_weights = class_weights / class_weights.sum() * len(class_counts) 
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32) 
    loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor) # measure prediction error (loss) 

    # note: this was the setting for the initial run
    # loss_function = nn.CrossEntropyLoss() # measure prediction error (loss) 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # update model weights based on loss (optimize)

    # NN status check
    print("Model created.") # print status label
    print(f"Input dimension: {input_dim}") # print number of input embedding dimensions
    print(f"Hidden dimension: {hidden_dim}") # print number of hidden dimensions
    print(f"Output classes: {output_dim}") # print number of topic categories
    
    dataset = TensorDataset(X_tensor, y_tensor) # organize feature and label tensors into PyTorch dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) # create dataloader for mini-batch training and randomized data order

    validation_accuracies = [] # added for bonus 1: list tracking validation accuracy

    # train loop for NN
    for epoch in range(args.epochs): # iterate over training epochs

        total_loss = 0 # track total loss

        for batch_X, batch_y in dataloader: # iterate through mini-batches of training data

            predictions = model(batch_X) # generate topic prediction scores from input embeddings
            loss = loss_function(predictions, batch_y) # measure prediction error against correct labels
            optimizer.zero_grad() # clear previous gradient calculations
            loss.backward() # compute gradients through backpropagation
            optimizer.step() # update model weights using computed gradients

            total_loss += loss.item() # add loss to total loss tracker

        average_loss = total_loss / len(dataloader) # compute average loss across all batches

        # added for bonus 1: 
        if X_dev_tensor is not None: # bonus 1: run validation evaluation if there is validation data
            with torch.no_grad(): # bonus 1: no gradient tracking during validation evaluation
                dev_predictions = model(X_dev_tensor) # bonus 1: generate prediction scores for validation embedding

            dev_predicted_labels = torch.argmax(dev_predictions, dim=1) # bonus 1: convert prediction scores into predicted class labels
            dev_accuracy = (dev_predicted_labels == y_dev_tensor).float().mean().item() # bonus 1: compute validation accuracy
            validation_accuracies.append(dev_accuracy) # bonus 1: add to validation accuracies tracking list

            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {average_loss}, Validation Accuracy: {dev_accuracy:.4f}") # bonus 1: print staus report

        else: # bonus 1: 
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {average_loss}") # bonus 1: fallback: print status report if no validation information

        # commented out after bonus 1 addition
        #print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {average_loss}") # print epoch progress and average training loss

    # added for bonus 1: plot of performance model
    if args.plot_output and validation_accuracies: # bonus 1: create plot if output path provided and validation accuracy data exists
        plt.figure() # bonus 1: create matplotlib figure
        plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies) # bonus 1: plot validation accuracy across epochs
        plt.xlabel("Epoch") # bonus 1: label x-axis
        plt.ylabel("Validation Accuracy") # bonus 1: label y-axis
        plt.title("Validation Accuracy by Epoch") # bonus 1: graph title
        plt.savefig(args.plot_output) # bonus 1: save graph as image
        plt.close() # bonus 1: close
        print("Validation accuracy plot saved to:") # bonus 1: print status label
        print(args.plot_output) # bonus 1: filepath

    torch.save(model.state_dict(), args.output_model) # save model to output path

    # model save status check
    print("Model saved to:") # print status label
    print(args.output_model) # print model filepath

# define command-line arguments
if __name__ == "__main__": # run only if this file is executed directly

    parser = argparse.ArgumentParser() # create parser for command-line arguments
    parser.add_argument("--train_tsv", type=str, required=True) # add required command-line path argument for training TSV
    parser.add_argument("--embeddings", type=str, required=True) # add required command-line path argument for embeddings file
    parser.add_argument("--dev_tsv", type=str, required=False) # added for bonus 1: validation TSV input
    parser.add_argument("--dev_embeddings", type=str, required=False) # added for bonus 1: validation embeddings input
    parser.add_argument("--output_model", type=str, required=True) # add required command-line path argument for model 
    parser.add_argument("--epochs", type=int, default=10) # add command-line argument for number of epochs (default to 10)
    parser.add_argument("--batch_size", type=int, default=32) # add command-line argument for batch size (default to 32)

    parser.add_argument("--plot_output", type=str, required=False) # added for bonus 1: add command-line argument for plot output
    
    args = parser.parse_args() # parse command-line input into usable Python variables

    main(args) # use command-line arguments to initiate classifier pipeline