import os
import statistics
from transformers import T5TokenizerFast
from tqdm import tqdm

# --- Configuration ---
# Ensure the 'data' folder is in the same directory as this script
DATA_DIR = "data"
TRAIN_NL_PATH = os.path.join(DATA_DIR, "train.nl")
TRAIN_SQL_PATH = os.path.join(DATA_DIR, "train.sql")
DEV_NL_PATH = os.path.join(DATA_DIR, "dev.nl")
DEV_SQL_PATH = os.path.join(DATA_DIR, "dev.sql")

TOKENIZER_NAME = "google-t5/t5-small"

# This is the prefix we defined in load_data.py
PREPROCESSING_PREFIX = "translate English to SQL: "
# ---

def load_lines(path):
    """Loads all lines from a file, stripping whitespace."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def analyze_split(nl_path, sql_path, tokenizer, nl_prefix=""):
    """
    Analyzes a given pair of .nl and .sql files.
    
    Args:
        nl_path (str): Path to the natural language file.
        sql_path (str): Path to the SQL file.
        tokenizer (T5TokenizerFast): The T5 tokenizer.
        nl_prefix (str): A prefix to add to every NL line (for Table 2).
    
    Returns:
        dict: A dictionary containing all required statistics.
    """
    print(f"Analyzing {nl_path} and {sql_path} (Prefix: '{nl_prefix}')...")
    nl_lines = load_lines(nl_path)
    sql_lines = load_lines(sql_path)

    if len(nl_lines) != len(sql_lines):
        raise ValueError(f"File line count mismatch: {nl_path} vs {sql_path}")

    num_examples = len(nl_lines)
    nl_token_lengths = []
    sql_token_lengths = []
    nl_vocab = set()
    sql_vocab = set()

    for i in tqdm(range(num_examples), desc="Tokenizing"):
        # Apply the prefix to the NL text
        nl_text = nl_prefix + nl_lines[i]
        sql_text = sql_lines[i]

        # Use .tokenize() to get subword strings for vocab counting
        nl_tokens = tokenizer.tokenize(nl_text)
        sql_tokens = tokenizer.tokenize(sql_text)
        
        # Use .encode() to get the true token length (including special tokens)
        nl_token_lengths.append(len(tokenizer.encode(nl_text)))
        sql_token_lengths.append(len(tokenizer.encode(sql_text)))
        
        # Update vocab sets
        nl_vocab.update(nl_tokens)
        sql_vocab.update(sql_tokens)

    # Calculate statistics
    stats = {
        "Number of examples": num_examples,
        "Mean sentence length (NL)": f"{statistics.mean(nl_token_lengths):.2f}",
        "Mean SQL query length": f"{statistics.mean(sql_token_lengths):.2f}",
        "Vocabulary size (NL)": len(nl_vocab),
        "Vocabulary size (SQL)": len(sql_vocab)
    }
    
    return stats

def print_table(title, train_stats, dev_stats):
    """Helper function to print a formatted table."""
    print("\n" + "=" * 62)
    print(f"{title}")
    print("=" * 62)
    
    # Get all keys from train_stats (order matters)
    headers = list(train_stats.keys())
    
    # Print header
    print(f"{'Statistics Name':<30} | {'Train':<15} | {'Dev':<15}")
    print("-" * 62)
    
    # Print data rows
    for key in headers:
        print(f"{key:<30} | {train_stats[key]:<15} | {dev_stats[key]:<15}")
    print("=" * 62)

def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_NAME)

    train_stats_raw = analyze_split(TRAIN_NL_PATH, TRAIN_SQL_PATH, tokenizer, nl_prefix="")
    dev_stats_raw = analyze_split(DEV_NL_PATH, DEV_SQL_PATH, tokenizer, nl_prefix="")

    train_stats_proc = analyze_split(TRAIN_NL_PATH, TRAIN_SQL_PATH, tokenizer, nl_prefix=PREPROCESSING_PREFIX)
    dev_stats_proc = analyze_split(DEV_NL_PATH, DEV_SQL_PATH, tokenizer, nl_prefix=PREPROCESSING_PREFIX)
    

    print_table("Table 1: Data statistics before any pre-processing",
                train_stats_raw,
                dev_stats_raw)
    
    train_stats_proc.pop("Number of examples")
    dev_stats_proc.pop("Number of examples")
    
    print_table("\nTable 2: Data statistics after pre-processing",
                train_stats_proc,
                dev_stats_proc)

if __name__ == "__main__":
    main()