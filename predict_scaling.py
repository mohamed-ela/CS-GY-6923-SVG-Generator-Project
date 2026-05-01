import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Data ---
MUP_RESULTS_FILE = 'mup_scaling_results.json'

def load_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def main():
    mup_results = load_results(MUP_RESULTS_FILE)
    if not mup_results:
        print(f"❌ Error: Could not find {MUP_RESULTS_FILE}")
        return

    # Sort results by parameter count
    mup_results = sorted(mup_results, key=lambda x: x['params'])
    
    # Extract data
    names = [r['name'] for r in mup_results]
    params = np.array([r['params'] for r in mup_results])
    losses = np.array([r['val_loss'] for r in mup_results])

    # --- 2. The Prediction Math ---
    # We will fit the line using ONLY Tiny, Small, Medium, and Large
    fit_params = params[:-1] 
    fit_losses = losses[:-1] 

    log_fit_params = np.log10(fit_params)
    log_fit_losses = np.log10(fit_losses)

    # Fit a linear polynomial (degree 1) to the log-log data
    slope, intercept = np.polyfit(log_fit_params, log_fit_losses, 1)
    alpha = -slope
    C = 10**intercept

    print("=== SCALING LAW PREDICTION ===")
    print(f"Calculated Alpha (from Tiny -> Large): {alpha:.4f}")
    print(f"Formula: Loss = {C:.4f} * (Params ^ -{alpha:.4f})\n")

    # --- 3. Evaluate the XL Prediction ---
    xl_actual_params = params[-1]
    xl_actual_loss = losses[-1]
    
    xl_predicted_loss = C * (xl_actual_params ** -alpha)
    error_margin = abs(xl_predicted_loss - xl_actual_loss) / xl_actual_loss * 100

    print(f"XL Actual Parameters:   {xl_actual_params:,}")
    print(f"XL Actual Loss:         {xl_actual_loss:.4f}")
    print(f"XL Predicted Loss:      {xl_predicted_loss:.4f}")
    print(f"Prediction Error:       {error_margin:.2f}%\n")

    # --- 4. Predict the 10x XL Model ---
    target_10x_params = xl_actual_params * 10
    target_10x_predicted_loss = C * (target_10x_params ** -alpha)
    print(f"🔮 10x XL Model (~{target_10x_params/1e6:.0f}M) Predicted Loss: {target_10x_predicted_loss:.4f}")

    # --- 5. Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot actual trained points
    plt.scatter(params, losses, color='blue', s=100, label='Actual Trained Models ($\mu$P)', zorder=5)
    for i, txt in enumerate(names):
        plt.annotate(txt, (params[i], losses[i]), textcoords="offset points", xytext=(0,15), ha='center')

    # Create trendline points (from Tiny up to 10x XL)
    trend_params = np.logspace(np.log10(params[0]), np.log10(target_10x_params), 100)
    trend_losses = C * (trend_params ** -alpha)

    # Plot the predicted trendline
    plt.plot(trend_params, trend_losses, linestyle='--', color='red', 
             label=f'Predicted Scaling Law ($\\alpha$={alpha:.4f})')
    
    # Mark the 10x XL prediction point
    plt.scatter(target_10x_params, target_10x_predicted_loss, color='red', marker='*', s=200, label='10x XL Prediction')
    plt.annotate('10x XL Model', (target_10x_params, target_10x_predicted_loss), textcoords="offset points", xytext=(0,15), ha='center', color='red')

    # Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.title('$\mu$P Scaling Law: Actuals vs Predictions', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Parameters (Log Scale)', fontsize=12)
    plt.ylabel('Validation Loss (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)

    plt.savefig('mup_scaling_prediction.png', dpi=300, bbox_inches='tight')
    print("\n🎉 Prediction plot saved as 'mup_scaling_prediction.png'")
    plt.show()

if __name__ == "__main__":
    main()