import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import mup
import cairosvg
import re
import shutil
from lxml import etree
from tokenizers import Tokenizer
from google.colab import drive

# --- 0. Drive Mounting ---
# Uncomment the line below if you haven't mounted your drive in this session yet
# drive.mount('/content/drive')

# --- 1. System Setup & Configurations ---
BASE_DIR = '/content/drive/MyDrive/SVG_Project/'
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'mup_XL_checkpoint.pth')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'svg_bpe_tokenizer.json')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data_splits/val.txt')
OUTPUT_DIR = 'generated_svgs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 512
VOCAB_SIZE = 4096

XL_CONFIG = {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072}

# --- 2. Model Architecture ---
class Head(nn.Module):
    def __init__(self, head_size, d_model):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scale = 1.0 / self.head_size
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_model) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        head_size = d_model // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, d_model)
        self.ffwd = FeedForward(d_model, d_ff)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x))
        return x

class SVG_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, config['d_model'])
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, config['d_model'])
        self.blocks = nn.Sequential(*[Block(config['d_model'], config['n_heads'], config['d_ff']) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.lm_head = mup.MuReadout(config['d_model'], VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        return logits

# --- 3. Loading Weights ---
print("Loading Tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print("Initializing XL Model with MuP shapes...")
model = SVG_GPT(XL_CONFIG).to(DEVICE)
proxy_base_model = SVG_GPT({"d_model": 128, "n_layers": 12, "n_heads": 12, "d_ff": 512}).to('cpu')
mup.set_base_shapes(model, proxy_base_model)

print(f"Loading weights from {CHECKPOINT_PATH}...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

# --- 4. Logic Functions ---

def clean_svg_output(raw_str):
    # Fixed sub-token spacing
    clean = raw_str.replace("< ", "<").replace(" >", ">").replace(" /", "/").replace(" =", "=").replace("= ", "=")
    # Numerical stitching (13 . 9 -> 13.9)
    clean = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', clean)
    # Attribute repair
    clean = clean.replace("stroke - width", "stroke-width").replace("stroke - opacity", "stroke-opacity")
    # Redundant tag repair
    if clean.count("<svg") > 1:
        clean = re.sub(r'<svg[^>]*>\s*<svg[^>]*>', '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">', clean)
    # Closure guarantee
    clean = clean.strip()
    if not clean.endswith("</svg>"):
        if 'd="' in clean and not clean.endswith('"/>'): clean += '"/>'
        clean += "</svg>"
    return clean

def generate_svg(model, tokenizer, prompt, max_new_tokens=512, temperature=0.8):
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -BLOCK_SIZE:]
            logits = model(x_cond)[:, -1, :] / temperature
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
            if "</svg>" in tokenizer.decode(x[0].tolist()): break
    return tokenizer.decode(x[0].tolist())

def evaluate_svg(svg_str, name):
    valid, rendered = False, False
    try:
        etree.fromstring(svg_str.encode('utf-8'))
        valid = True
    except: pass
    try:
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=os.path.join(OUTPUT_DIR, f"{name}.png"))
        rendered = True
    except: pass
    with open(os.path.join(OUTPUT_DIR, f"{name}.svg"), "w") as f: f.write(svg_str)
    return {"xml_valid": valid, "renders": rendered}

# --- 5. Final Execution Block ---
if __name__ == '__main__':
    print(f"🚀 FINAL PROJECT RUN (DEVICE: {DEVICE} | TEMP: 0.8)")

    conditional_prompts = {
        "face_start": '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" stroke="black" fill="none"/>',
        "box_start": '<svg viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" stroke="blue" ',
        "path_red": '<svg viewBox="0 0 24 24"><path fill="red" d="M12 2L15 8H21',
        "multi_circle": '<svg viewBox="0 0 100 100"><circle cx="30" cy="30" r="10"/><circle cx="70" cy="30" r="10"',
        "green_line": '<svg viewBox="0 0 100 100"><path stroke="green" d="M0 50 L100 50 M50 0 L50 100'
    }

    results = []
    success_count = 0

    # 2 attempts per prompt to aim for 5+ successes
    for attempt in range(2):
        for name, prefix in conditional_prompts.items():
            print(f"  Generating: {name} (Attempt {attempt+1})...")
            raw_out = generate_svg(model, tokenizer, prefix, temperature=0.8)
            clean_out = clean_svg_output(raw_out)
            metrics = evaluate_svg(clean_out, f"final_{name}_{attempt}")
            results.append(metrics)
            if metrics['renders']:
                success_count += 1
                print("    ✅ Success!")
            else:
                print("    ❌ Syntax Error")

    print(f"\n📊 Calculating Perplexity (Validation Set)...")
    with open(TEST_DATA_PATH, 'r') as f:
        test_data = f.read(100_000)

    tokens = tokenizer.encode(test_data).ids
    data = torch.tensor(tokens, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        xb = data[:BLOCK_SIZE].unsqueeze(0)
        yb = data[1:BLOCK_SIZE+1].unsqueeze(0)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1)).item()
        perp = math.exp(loss)

    print("\n" + "="*40)
    print(f"🏆 FINAL PROJECT METRICS")
    print(f"Model Perplexity: {perp:.4f}")
    print(f"Total Attempts:   {len(results)}")
    print(f"Total Rendered:   {success_count}")
    print("="*40)

    # Backup to Drive
    drive_dest = os.path.join(BASE_DIR, 'final_project_results')
    shutil.copytree(OUTPUT_DIR, drive_dest, dirs_exist_ok=True)
    print(f"📂 Results backed up to: {drive_dest}")