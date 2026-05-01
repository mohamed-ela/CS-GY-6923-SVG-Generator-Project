import os
import re
import random
import lxml.etree as etree
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# --- Configuration ---
VOCAB_SIZE = 4096
MAX_TOKENS = 1024
TARGET_TRAIN_TOKENS = 100_000_000

# Set to False to run the full dataset
DEV_MODE = False 

DATASETS_TO_LOAD = [
    "starvector/svg-icons-simple",
    "starvector/svg-emoji-simple",
    "starvector/svg-stack-simple"
]

# --- 1. Normalization and Validation ---
def clean_and_normalize_svg(svg_code):
    """Strips comments, normalizes coordinates, and protects XML structure."""
    # Protect the XML version string from the decimal rounding regex
    svg_code = svg_code.replace('version="1.0"', 'TEMP_VERSION_PLACEHOLDER')
    
    # Strip XML comments
    svg_code = re.sub(r'', '', svg_code, flags=re.DOTALL)
    
    # Normalize coordinate precision (round to 1 decimal place)
    def round_match(match):
        val = float(match.group(0))
        formatted = f"{val:.1f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"
        
    svg_code = re.sub(r'-?\d+\.\d+', round_match, svg_code)
    
    # Restore the XML version string
    svg_code = svg_code.replace('TEMP_VERSION_PLACEHOLDER', 'version="1.0"')
    
    # Strip unnecessary whitespace
    svg_code = re.sub(r'\s+', ' ', svg_code)
    svg_code = re.sub(r'>\s+<', '><', svg_code)
    return svg_code.strip()

def validate_svg(svg_code):
    """Parses, repairs missing quotes, and returns strictly valid XML."""
    if len(svg_code) < 50:
        return None
    
    try:
        # StarVector removes quotes to save space. recover=True allows lxml to fix this.
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(svg_code.encode('utf-8'), parser=parser)
        
        # Convert the repaired tree BACK into a strictly valid, properly quoted XML string
        valid_xml_string = etree.tostring(root, encoding='unicode')
        valid_xml_string = valid_xml_string.replace('<?xml version="1.0"?>\n', '')
        
        return valid_xml_string.strip()
    except Exception:
        return None

# --- 2. Data Collection Pipeline ---
def download_and_clean_data():
    raw_count = 0
    valid_svgs = []
    
    for ds_name in DATASETS_TO_LOAD:
        print(f"\nLoading dataset: {ds_name}")
        try:
            ds = load_dataset(ds_name, split="train")
            
            # This safely handles the smaller emoji dataset without throwing an Out Of Range error!
            if DEV_MODE: 
                ds = ds.select(range(min(5000, len(ds))))
            
            raw_count += len(ds)
            
            for item in tqdm(ds, desc=f"Cleaning {ds_name}"):
                raw_code = None
                
                # Dynamically find the SVG code no matter what the column is named
                for val in item.values():
                    if isinstance(val, str) and "<svg" in val:
                        raw_code = val
                        break
                
                if not raw_code: 
                    continue
                
                cleaned = clean_and_normalize_svg(raw_code)
                validated_and_repaired = validate_svg(cleaned)
                
                if validated_and_repaired:
                    valid_svgs.append(validated_and_repaired)
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            
    return valid_svgs, raw_count

# --- 3. Tokenizer Training ---
def train_bpe_tokenizer(svg_list):
    print("\nTraining BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, 
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )
    
    def batch_iterator():
        for i in range(0, len(svg_list), 1000):
            yield svg_list[i : i + 1000]
            
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    tokenizer.save("svg_bpe_tokenizer.json")
    print(f"Tokenizer saved with vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("=== Phase 1: Data Download & Cleaning ===")
    cleaned_svgs, total_raw_files = download_and_clean_data()
    print(f"Files after initial cleaning: {len(cleaned_svgs)} / {total_raw_files}")

    if not cleaned_svgs:
        print("❌ ERROR: 0 SVGs passed validation. Exiting to prevent crash.")
        exit()

    print("\n=== Phase 2: Tokenization ===")
    tokenizer = train_bpe_tokenizer(cleaned_svgs)

    print("\n=== Phase 3: Token Threshold Filtering ===")
    final_svgs = []
    token_lengths = []
    
    for svg in tqdm(cleaned_svgs, desc="Tokenizing and Filtering"):
        encoded = tokenizer.encode(svg)
        num_tokens = len(encoded.tokens)
        
        if num_tokens <= MAX_TOKENS:
            final_svgs.append(svg)
            token_lengths.append(num_tokens)

    if not final_svgs:
        print("❌ ERROR: 0 SVGs survived the max token threshold. Exiting.")
        exit()

    print("\n=== Phase 4: Splitting Data ===")
    random.seed(42)
    combined = list(zip(final_svgs, token_lengths))
    random.shuffle(combined)
    final_svgs, token_lengths = zip(*combined)

    total_files = len(final_svgs)
    train_idx = int(total_files * 0.98)
    val_idx = train_idx + int(total_files * 0.01)

    train_data = final_svgs[:train_idx]
    val_data = final_svgs[train_idx:val_idx]
    test_data = final_svgs[val_idx:]

    train_tokens = sum(token_lengths[:train_idx])
    val_tokens = sum(token_lengths[train_idx:val_idx])
    test_tokens = sum(token_lengths[val_idx:])

    os.makedirs("data_splits", exist_ok=True)
    with open("data_splits/train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
    with open("data_splits/val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))
    with open("data_splits/test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_data))

    print("\n=== Phase 5: Statistics for your Report ===")
    print(f"Total files BEFORE filtering: {total_raw_files}")
    print(f"Total files AFTER filtering:  {total_files}")
    print(f"Vocabulary Size:              {tokenizer.get_vocab_size()}")
    print("-------------------------------------------------")
    print(f"Train Tokens: {train_tokens:,} ({len(train_data):,} files)")
    print(f"Val Tokens:   {val_tokens:,} ({len(val_data):,} files)")
    print(f"Test Tokens:  {test_tokens:,} ({len(test_data):,} files)")
    print("-------------------------------------------------")

    print("\nGenerating Sequence Length Histogram...")
    plt.hist(token_lengths, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.title("SVG Sequence Length Distribution (Tokens)")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("sequence_length_histogram.png")
    print("Saved histogram to 'sequence_length_histogram.png'.")

    print("\nSaving raw .svg examples for your report...")
    os.makedirs("examples", exist_ok=True)
    sorted_by_len = sorted(combined, key=lambda x: x[1])
    
    examples_to_save = {
        "simple": sorted_by_len[10][0],               
        "medium": sorted_by_len[len(sorted_by_len)//2][0], 
        "complex": sorted_by_len[-10][0]              
    }
    
    for complexity, svg_str in examples_to_save.items():
        file_path = f"examples/{complexity}_example.svg"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_str)
            
    print("Saved simple, medium, and complex SVGs to the 'examples/' folder.")
    print("\n✅ Part 1 Complete!")