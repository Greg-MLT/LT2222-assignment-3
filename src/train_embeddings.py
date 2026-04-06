import argparse
import pandas as pd

def main(args):
    print("Training embeddings;")
    print("Input file:", args.input)
    print("Output file:", args.output)

    data = pd.read_csv(args.input, sep="\t")

    print("First few sentences:")
    print(data["text"].head())

    print("\nFirst few labels:")
    print(data["category"].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="Path to input file")
    parser.add_argument("--output", type=str, help="Path to output file")
    
    args = parser.parse_args()
    
    main(args)
