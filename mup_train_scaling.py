import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import mup
from google.colab import drive
from tokenizers import Tokenizer

# --- 1. System Setup & Drive Mounting ---
drive.mount('/content/drive')
BASE_DIR = '/content/drive/MyDrive/SVG_Project/' # CHANGE THIS if your path is different!
os.makedirs(BASE_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(BASE_DIR, 'mup_scaling_results.json')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"USING DEVICE: {DEVICE}")

# --- 2. Hyperparameters & Configurations ---
BLOCK_SIZE = 512
BATCH_SIZE = 8
VOCAB_SIZE = 4096 # Matches Part 2 setup!
TARGET_TOKENS = 100_000_000 # 100M tokens

# --- 3. Data Loading ---
def load_data(split, tokenizer, is_train=False):
    print(f"Loading {split} data into memory (chunked to save RAM)...")
    file_path = os.path.join(BASE_DIR, f"data_splits/{split}.txt")
    token_limit = TARGET_TOKENS if is_train else 500_000
    all_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(10_000_000)
            if not chunk:
                break
            encoded = tokenizer.encode(chunk)
            all_ids.extend(encoded.ids)
            if len(all_ids) >= token_limit:
                break
    all_ids = all_ids[:token_limit]
    data = torch.tensor(all_ids, dtype=torch.long)
    print(f"✅ Loaded {len(data):,} tokens for {split} split.")
    return data

def get_batch(data):
    """Fetches a random batch of X and Y from the provided data tensor."""
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def estimate_loss(model, data, eval_iters=100):
    """Estimates validation loss cleanly."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data)
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=='cuda')):
                _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

# --- 4. MuP Architecture ---
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
        # MuP requirement: scale attention by 1/d instead of 1/sqrt(d)
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
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        head_size = d_model // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, d_model)
        self.ffwd = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SVG_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, config['d_model'])
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, config['d_model'])
        self.blocks = nn.Sequential(*[Block(config['d_model'], config['n_heads'], config['d_ff']) for _ in range(config['n_layers'])])
        self.ln_f = nn.LayerNorm(config['d_model'])
        # MuP requirement: Use MuReadout for the final layer
        self.lm_head = mup.MuReadout(config['d_model'], VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- 5. Training Logic with Checkpointing & MuP Optimizer ---
def train_model(name, config, lr, train_data, val_data, base_model):
    print(f"\n=== Training {name} Model (MuP) ===")

    # Initialize the target model
    model = SVG_GPT(config).to(DEVICE)

    # MuP REQUIREMENT: Calculate scaling factors using the proxy base_model
    mup.set_base_shapes(model, base_model)

    # MuP REQUIREMENT: Use MuAdamW
    optimizer = mup.MuAdamW(model.parameters(), lr=lr)

    num_params = model.get_num_params()
    print(f"Parameters: {num_params:,}")

    tokens_per_batch = BATCH_SIZE * BLOCK_SIZE
    total_steps = TARGET_TOKENS // tokens_per_batch
    print(f"Target steps for 100M tokens: {total_steps:,}")

    start_time = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = None
    if DEVICE == 'cuda':
        scaler = torch.amp.GradScaler('cuda')

    start_step = 0
    checkpoint_path = os.path.join(BASE_DIR, f"mup_{name}_checkpoint.pth")

    # Crash Recovery Logic
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found saved checkpoint for {name}! Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        for _ in range(start_step):
            scheduler.step()
        print(f"Resuming from step {start_step} / {total_steps}...")
    else:
        print("Starting fresh from step 0.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for step in range(start_step, total_steps):
        X, Y = get_batch(train_data)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=='cuda')):
            _, loss = model(X, Y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        if step % 500 == 0:
            print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f}")

        # Checkpointing
        if step > 0 and step % 2000 == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"💾 Checkpoint saved at step {step}!")

    end_time = time.time()
    val_loss = estimate_loss(model, val_data)

    wall_time = end_time - start_time
    throughput = ((total_steps - start_step) * tokens_per_batch) / wall_time if wall_time > 0 else 0
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"Finished {name}! Val Loss: {val_loss:.4f} | Mem: {gpu_mem:.0f}MB")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("🗑️ Cleaned up temporary checkpoint file.")

    return {
        "name": name,
        "params": num_params,
        "val_loss": val_loss,
        "wall_time_sec": wall_time,
        "peak_gpu_mem_mb": gpu_mem,
        "tokens_per_sec": throughput
    }

# --- 6. Execution Pipeline ---
if __name__ == '__main__':
    # Load Data at the start of Section 6!
    tokenizer_path = os.path.join(BASE_DIR, "svg_bpe_tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    train_data = load_data("train", tokenizer, is_train=True)
    val_data = load_data("val", tokenizer, is_train=False)

    # 6a. Apply Your Optimal Learning Rate
    # MAKE SURE THIS MATCHES YOUR WINNING SWEEP RESULT
    OPTIMAL_MUP_LR = 0.003
    print(f"Selected Optimal MuP LR: {OPTIMAL_MUP_LR}")

    # 6b. Train All Remaining Models
    print("\n--- Starting Full Scale Training ---")

    # Load existing results if they exist to avoid overwriting
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        completed_models = [r['name'] for r in results]
    else:
        results = []
        completed_models = []

    # NEW DICTIONARY: Notice "Tiny" is skipped, and Small n_heads is fixed to 8
    MODELS_TO_TRAIN = {
        "Tiny":   {"d_model": 128, "n_layers": 4,  "n_heads": 4,  "d_ff": 512},
        "Small":  {"d_model": 256, "n_layers": 6,  "n_heads": 8,  "d_ff": 1024},
        "Medium": {"d_model": 384, "n_layers": 8,  "n_heads": 8,  "d_ff": 1536},
        "Large":  {"d_model": 512, "n_layers": 10, "n_heads": 8,  "d_ff": 2048},
        "XL":     {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072}
    }

    for name, config in MODELS_TO_TRAIN.items():
        if name in completed_models:
            print(f"Skipping {name}, already trained in JSON.")
            continue

        # -------------------------------------------------------------
        # MUP FIX: Create a custom proxy base model for each target
        # It takes the target's n_layers/n_heads, but the Tiny model's width
        # -------------------------------------------------------------
        proxy_base_config = config.copy()
        proxy_base_config['d_model'] = 128  # Force Tiny Width
        proxy_base_config['d_ff'] = 512     # Force Tiny Width
        proxy_base_model = SVG_GPT(proxy_base_config).to('cpu')

        # Pass the newly generated proxy_base_model to the train function
        metrics = train_model(name, config, OPTIMAL_MUP_LR, train_data, val_data, proxy_base_model)
        results.append(metrics)

        # Save securely after every model finishes
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    print("\n✅ All MuP models trained successfully! Results saved to:", RESULTS_FILE)