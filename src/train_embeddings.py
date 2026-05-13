"""
--Part 2: Sentence Embeddings--
This script takes TSV files of Chinese sentences and their topic labels, trains 
FastText character-level embeddings on the text, and converts each sentence 
into a fixed-length vector (sentence embedding) by averaging the
character embeddings.  The script outputs embedding files 
containing one sentence embedding per sentence.
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
    print("Training embeddings;") # print status label
    print("Input files:", args.inputs) # print list of input TSV files
    print("Output files:", args.outputs) # print list of output embedding files

    ## debug: a look at the data file
    #print("First few sentences:")
    #print(data["text"].head()) # demonstrate first few rows from 'text' column

    #print("\nFirst few labels:")
    #print(data["category"].head()) # show first few topic labels
    
    #print("\nFirst sentence as characters:")
    #print(list(data["text"].iloc[0])) # show first sentence as characters

    all_sentences = [] # store sentences from all datasets for shared FastText training

    datasets = [] # store each dataset separately for later embedding generation

    # load all TSV files and collect sentences
    for input_file in args.inputs: # iterate through all input TSV files

        data = pd.read_csv(input_file, sep="\t") # load TSV file into pandas table
        sentences = data["text"].apply(list) # convert each sentence into list of characters
        datasets.append((input_file, sentences)) # store filename and tokenized sentences together
        all_sentences.extend(sentences.tolist()) # add sentences to combined corpus for shared FastText model

    print("\nTraining FastText model...") # status update: initiating training of model

    model = FastText(
        sentences=all_sentences, # sentences as lists of characters
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

    ## debug: print samples
    #print("\nFirst 3 sentence embeddings:")
    #print(all_embeddings.head(3)) # print first 3 sentence embeddings

    #print("\nVector for one character:")
    #print(model.wv[sentences.iloc[0][0]]) # print vector for a single character

    #print("\nFirst 3 sentences as character lists:")
    #print(sentences.head(3)) # print first 3 sentences as character lists

    # create embeddings separately for each dataset using shared model
    for (input_file, sentences), output_file in zip(datasets, args.outputs): # pair each dataset with its corresponding output file

        print(f"\nCreating embeddings for: {input_file}") # print current dataset being processed
        all_embeddings = sentences.apply(lambda s: sentence_to_embedding(s, model)) # convert all sentences into averaged sentence embeddings
        embedding_df = pd.DataFrame(all_embeddings.tolist()) # convert sentence embeddings into tabular format
        embedding_df.to_csv(output_file, sep="\t", index=False) # save embeddings table to output TSV file

        print("Embeddings saved to:") # print status label
        print(os.path.abspath(output_file)) # print absolute filepath of saved embeddings file

if __name__ == "__main__": # if current file is executed directly, run this block
    parser = argparse.ArgumentParser() # interpret command line input
    
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to input TSV files") # command-line path to input TSV file(s)
    parser.add_argument("--outputs", nargs="+", required=True, help="Paths to output embedding files") # command-line path where embeddings file(s) will be saved
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension") # number of dimensions in each embedding vector
    
    args = parser.parse_args() # read command line and convert to usable variables (see above)

    # guard
    if len(args.inputs) != len(args.outputs): # verify that each input file has a matching output file
        raise ValueError("Number of input and output files must match.") # stop script if counts do not align
    
    main(args) # function call to 'main' with 'args' as argument
