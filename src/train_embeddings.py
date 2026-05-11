"""
--Part 2: Sentence Embeddings--
This script takes a TSV file of Chinese sentences and their topic labels,
trains FastText character-level embeddings on the text, and converts each
sentence into a fixed-length vector (sentence embedding) by averaging the
character embeddings.  The script outputs a file containing one embedding per sentence.
"""
import argparse # handles command line arguments
import pandas as pd # manage table data
from gensim.models import FastText # learn embeddings
import numpy as np # math: compute mean of token-level vectors into sentence-level representation
import os # handle file paths

# Convert a tokenized sentence (list of characters) into a sentence embedding using FastText
def sentence_to_embedding(sentence, model): # input sentence (split into characters), FastText model
    vectors = [model.wv[char] for char in sentence] # iterate over each character and retrieve its character-level vector
    return np.mean(vectors, axis=0) # average character embeddings for sentence-level vector

def main(args):

    # print status report
    print("Training embeddings;")
    print("Input file:", args.input)
    print("Output file:", args.output)

    data = pd.read_csv(args.input, sep="\t") # load TSV file into pandas table

    ## debug: a look at the data file
    #print("First few sentences:")
    #print(data["text"].head()) # demonstrate first few rows from 'text' column

    #print("\nFirst few labels:")
    #print(data["category"].head()) # show first few topic labels
    
    #print("\nFirst sentence as characters:")
    #print(list(data["text"].iloc[0])) # show first sentence as characters

    sentences = data["text"].apply(list) # from 'text' column in 'data' convert each sentence to list of characters

    print("\nTraining FastText model...") # status update: initiating training of model

    model = FastText(
        sentences=sentences, # sentences as lists of characters
        vector_size=args.dim, # number of dimensions per vector; dim set in command line
        window=3, # context is +/- 3 positions
        min_count=1, # keeps all characters
        epochs=5 # 5 iterations over the data for training
    )
    
    print("Model trained.") # status update: training completed

    ## debug: a look at embedding of first sentence
    #print("\nFirst sentence embedding:") # first sentence status update
    #first_sentence = sentences.iloc[0] # select list of first sentence characters
    #sentence_embedding = sentence_to_embedding(first_sentence, model) # compute first sentence embedding by averaging character vectors
    #print(sentence_embedding) # print sentence embedding

    # calculate sentence embeddings and report status
    print("\nCreating embeddings for all sentences...") # status update: calculating embeddings
    all_embeddings = sentences.apply(lambda s: sentence_to_embedding(s, model)) # calculate all sentence embeddings
    print("Done.") # status update: embeddings calculated

    ## debug: print samples
    #print("\nFirst 3 sentence embeddings:")
    #print(all_embeddings.head(3)) # print first 3 sentence embeddings

    #print("\nVector for one character:")
    #print(model.wv[sentences.iloc[0][0]]) # print vector for a single character

    #print("\nFirst 3 sentences as character lists:")
    #print(sentences.head(3)) # print first 3 sentences as character lists

    embedding_df = pd.DataFrame(all_embeddings.tolist()) # convert sentence embeddings to table for later use; rows = sentences, columns embedding dimensions

    embedding_df.to_csv(args.output, sep="\t", index=False) # save table (embedding_df) to file; path designated in command line

    print("\nEmbeddings saved to:") # status: saved file location
    print(os.path.abspath(args.output)) # print user-established filepath

if __name__ == "__main__": # if current file is executed directly, run this block
    parser = argparse.ArgumentParser() # interpret command line input
    
    parser.add_argument("--input", type=str, required=True, help="Path to input file") # command-line path to input TSV file
    parser.add_argument("--output", type=str, required=True, help="Path to output file") # command-line path where embeddings will be saved
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension") # number of dimensions in each embedding vector
    
    args = parser.parse_args() # read command line and convert to usable variables (see above)
    
    main(args) # function call to 'main' with 'args' as argument
