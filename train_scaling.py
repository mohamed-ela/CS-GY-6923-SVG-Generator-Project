import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# --- 1. Configurations & Hyperparameters ---
BLOCK_SIZE = 512 # Max context length
VOCAB_SIZE = 4096
BATCH_SIZE = 8    # Adjust if hit Out Of Memory (OOM) errors!
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#Learning Rate 0.003
# Model configurations specified in the assignment
MODEL_CONFIGS = {
    "Tiny":   {"d_model": 128, "n_layers": 4,  "n_heads": 4,  "d_ff": 512},
    "Small":  {"d_model": 192, "n_layers": 6,  "n_heads": 6,  "d_ff": 768},
    "Medium": {"d_model": 384, "n_layers": 6,  "n_heads": 6,  "d_ff": 1536},
    "Large":  {"d_model": 512, "n_layers": 10,  "n_heads": 8,  "d_ff": 2048},
    "XL":     {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072}

}

# --- 2. Decoder-Only Transformer (nanoGPT style) ---
class Head(nn.Module):
    def __init__(self, head_size, d_model):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, d_model) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
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
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, config["d_model"])
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, config["d_model"])
        self.blocks = nn.Sequential(*[Block(config["d_model"], config["n_heads"], config["d_ff"]) for _ in range(config["n_layers"])])
        self.ln_f = nn.LayerNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], VOCAB_SIZE)

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

# --- 3. Data Loading ---
def load_data(split, tokenizer):
    with open(f"data_splits/{split}.txt", "r", encoding="utf-8") as f:
        text = f.read()
   
    # This is a simplified in-memory load for demonstration
    encoded = tokenizer.encode(text[:5000000]) # Truncated to prevent RAM explosion during testing
    data = torch.tensor(encoded.ids, dtype=torch.long)
    return data

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, data, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()



# --- 5. Main Training Loop ---
def train_model(name, config, lr, train_data, val_data):
    print(f"\n=== Training {name} Model ===")
    model = SVG_GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_params = model.get_num_params()
    print(f"Parameters: {num_params:,}")

    # Calculate steps for exactly 1 epoch
    tokens_per_batch = BATCH_SIZE * BLOCK_SIZE
    total_steps = len(train_data) // tokens_per_batch
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Cosine learning rate scheduler setup would go here
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    for step in range(total_steps):
        X, Y = get_batch(train_data)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f}")

    end_time = time.time()
    val_loss = estimate_loss(model, val_data)
    
    wall_time = end_time - start_time
    throughput = (total_steps * tokens_per_batch) / wall_time
    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    print(f"Finished {name}! Val Loss: {val_loss:.4f} | Time: {wall_time:.0f}s | Mem: {gpu_mem:.0f}MB | Tok/sec: {throughput:.0f}")

    return {
        "name": name,
        "params": num_params,
        "val_loss": val_loss,
        "wall_time_sec": wall_time,
        "peak_gpu_mem_mb": gpu_mem,
        "tokens_per_sec": throughput
    }

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("svg_bpe_tokenizer.json")
    train_data = load_data("train", tokenizer)
    val_data = load_data("val", tokenizer)

    best_lr = 0.003
    
    results = []
    for name, config in MODEL_CONFIGS.items():
        metrics = train_model(name, config, best_lr, train_data, val_data)
        results.append(metrics)
        
        # Save after every model in case of a crash!
        with open("scaling_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    print("\n✅ All models trained! Results saved to 'scaling_results.json'.")