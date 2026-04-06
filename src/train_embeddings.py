import argparse

def main():
    print("Training embeddings...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="Path to input file")
    parser.add_argument("--output", type=str, help="Path to output file")
    
    args = parser.parse_args()
    
    main()
