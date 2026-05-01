import os
import json
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configurations ---
STANDARD_RESULTS_FILE = 'scaling_results.json'

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
    print("Loading standard scaling results...")
    std_results = load_results(STANDARD_RESULTS_FILE)

    if not std_results:
        print(f"❌ Error: Could not find {STANDARD_RESULTS_FILE} in the current directory.")
        return

    plt.figure(figsize=(10, 6))

    # --- 2. Process Standard Models ---
    # Sort by parameters to ensure the line connects in the right order
    std_results = sorted(std_results, key=lambda x: x['params'])
    std_params = [r['params'] for r in std_results]
    std_losses = [r['val_loss'] for r in std_results]
    std_names = [r['name'] for r in std_results]
    
    # Calculate Alpha
    std_alpha = calculate_alpha(std_params, std_losses)
    label_text = f'Standard SP | $\\alpha$ = {std_alpha:.4f}'

    # --- 3. Plotting ---
    plt.plot(std_params, std_losses, marker='o', linestyle='-', color='red', 
             label=label_text, linewidth=2, markersize=8)
    
    # Annotate each point with the model name (Tiny, Small, etc.)
    for i, name in enumerate(std_names):
        plt.annotate(name, (std_params[i], std_losses[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', color='red')

    # --- 4. Chart Formatting ---
    # Both axes must be log scale to accurately visualize power-law scaling
    plt.xscale('log')
    plt.yscale('log') 
    
    plt.title('Validation Loss Scaling: Standard SP (Part 2)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Parameters (Log Scale)', fontsize=12)
    plt.ylabel('Validation Loss (Log Scale)', fontsize=12)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    # --- 5. Save and Display ---
    output_filename = 'standard_scaling_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"🎉 Plot saved successfully as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()