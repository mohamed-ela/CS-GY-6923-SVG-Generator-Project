import os
import json
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configurations ---
STANDARD_RESULTS_FILE = 'scaling_results.json' # From Part 2
MUP_RESULTS_FILE = 'mup_scaling_results.json'  # From Part 3

def load_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def calculate_alpha(params, losses):
    """
    Calculates the scaling exponent alpha (slope) using log-log linear regression.
    Formula: log(Loss) = -alpha * log(Params) + log(C)
    """
    log_params = np.log10(params)
    log_losses = np.log10(losses)
    
    # polyfit returns [slope, intercept]
    slope, intercept = np.polyfit(log_params, log_losses, 1)
    
    # Alpha is the negative of the slope
    alpha = -slope 
    return alpha

def main():
    print("Loading scaling results...")
    std_results = load_results(STANDARD_RESULTS_FILE)
    mup_results = load_results(MUP_RESULTS_FILE)

    if not std_results and not mup_results:
        print("❌ Error: Could not find any JSON result files in the current directory.")
        return

    plt.figure(figsize=(10, 6))

    # --- 2. Plot Standard Models (Part 2) ---
    if std_results:
        std_results = sorted(std_results, key=lambda x: x['params'])
        std_params = [r['params'] for r in std_results]
        std_losses = [r['val_loss'] for r in std_results]
        std_names = [r['name'] for r in std_results]
        
        # Calculate Alpha
        std_alpha = calculate_alpha(std_params, std_losses)
        label_text = f'Standard SP (Part 2) | $\\alpha$ = {std_alpha:.4f}'

        plt.plot(std_params, std_losses, marker='o', linestyle='-', color='red', 
                 label=label_text, linewidth=2, markersize=8)
        
        for i, name in enumerate(std_names):
            plt.annotate(name, (std_params[i], std_losses[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', color='red')

    # --- 3. Plot MuP Models (Part 3) ---
    if mup_results:
        mup_results = sorted(mup_results, key=lambda x: x['params'])
        mup_params = [r['params'] for r in mup_results]
        mup_losses = [r['val_loss'] for r in mup_results]
        mup_names = [r['name'] for r in mup_results]

        # Calculate Alpha
        mup_alpha = calculate_alpha(mup_params, mup_losses)
        label_text = f'MuP (Part 3) | $\\alpha$ = {mup_alpha:.4f}'

        plt.plot(mup_params, mup_losses, marker='s', linestyle='-', color='blue', 
                 label=label_text, linewidth=2, markersize=8)
        
        for i, name in enumerate(mup_names):
            plt.annotate(name, (mup_params[i], mup_losses[i]), 
                         textcoords="offset points", xytext=(0,-18), ha='center', color='blue')

    # --- 4. Chart Formatting ---
    # Both axes must be log scale to accurately visualize power-law scaling
    plt.xscale('log')
    plt.yscale('log') 
    
    plt.title('Validation Loss Scaling: Standard SP vs $\mu$P', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Parameters (Log Scale)', fontsize=12)
    plt.ylabel('Validation Loss (Log Scale)', fontsize=12)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    # --- 5. Save and Display ---
    output_filename = 'final_scaling_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"🎉 Plot saved successfully as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()