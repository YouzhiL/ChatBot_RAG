import argparse
from base_parser import run_base_parse
from llama_parser import run_llama_parse

if __name__ == "__main__":
    # Set up argument parsing for CLI
    parser = argparse.ArgumentParser(description="Document Query CLI with Parsing Options.")
    
    # Add a choice for selecting the parser type
    parser.add_argument(
        "parser_type",
        choices=["base_parse", "llama_parse"],
        help="Choose between 'base_parse' or 'llama_parse'."
    )

    # Add an argument for file paths
    parser.add_argument(
        "files",
        nargs="+",  # Accepts multiple file paths
        type=str,
        help="File paths of your customized data put in ./data folder.(e.g., 'data/doc1.pdf data/doc2.pdf')."
    )

    # Parse the provided arguments
    args = parser.parse_args()

    # Run the selected parser with the provided file paths
    if args.parser_type == "base_parse":
        run_base_parse(args.files)
    elif args.parser_type == "llama_parse":
        run_llama_parse(args.files)