import argparse
import pandas as pd
from gensim.models import FastText
import numpy as np
import os

def sentence_to_embedding(sentence, model):
    vectors = [model.wv[char] for char in sentence]
    return np.mean(vectors, axis=0)

def main(args):
    print("Training embeddings;")
    print("Input file:", args.input)
    print("Output file:", args.output)

    data = pd.read_csv(args.input, sep="\t")

    print("First few sentences:")
    print(data["text"].head())

    print("\nFirst few labels:")
    print(data["category"].head())

    print("\nFirst sentence as characters:")
    print(list(data["text"].iloc[0]))

    sentences = data["text"].apply(list)

    print("\nTraining FastText model...")

    model = FastText(
        sentences=sentences,
        vector_size=args.dim,
        window=3,
        min_count=1,
        epochs=5
    )
    
    print("Model trained.")

    print("\nFirst sentence embedding:")

    first_sentence = sentences.iloc[0]
    sentence_embedding = sentence_to_embedding(first_sentence, model)

    print(sentence_embedding)

    print("\nCreating embeddings for all sentences...")

    all_embeddings = sentences.apply(lambda s: sentence_to_embedding(s, model))

    print("Done.")

    print("\nFirst 3 sentence embeddings:")
    print(all_embeddings.head(3))

    print("\nVector for one character:")
    print(model.wv[sentences.iloc[0][0]])

    print("\nFirst 3 sentences as character lists:")
    print(sentences.head(3))

    embedding_df = pd.DataFrame(all_embeddings.tolist())

    embedding_df.to_csv(args.output, sep="\t", index=False)

    print("\nEmbeddings saved to:")
    print(os.path.abspath(args.output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="Path to input file")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension")
    
    args = parser.parse_args()
    
    main(args)
