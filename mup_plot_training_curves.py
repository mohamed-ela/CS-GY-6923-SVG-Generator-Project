import re
import os
import matplotlib.pyplot as plt

# --- 1. Configurations ---
LOG_FILE = 'Mup_Training_Loss.txt'

def main():
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: {LOG_FILE} not found in the current directory.")
        return

    # Dictionary to store our parsed data
    training_data = {}
    current_model = None

    # --- 2. Robust Regex Parsing ---
    # Matches "Training Tiny Model (MuP)", "Training XL Model (MuP)", etc.
    model_pattern = re.compile(r'Training\s+([A-Za-z]+)\s+Model')
    
    # Matches lines like: "Step 500/24414 | Loss: 2.0760"
    step_pattern = re.compile(r'Step\s+(\d+)/\d+\s+\|\s+Loss:\s+([0-9.]+)')

    print(f"Parsing {LOG_FILE}...")
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. Check if the line is a model header
            model_match = model_pattern.search(line)
            if model_match:
                current_model = model_match.group(1)
                if current_model not in training_data:
                    training_data[current_model] = {'steps': [], 'losses': []}
                print(f"✅ Found model: {current_model}")
                continue
            
            # 2. Check if the line is a step/loss log
            if current_model:
                step_match = step_pattern.search(line)
                if step_match:
                    step = int(step_match.group(1))
                    loss = float(step_match.group(2))
                    training_data[current_model]['steps'].append(step)
                    training_data[current_model]['losses'].append(loss)

    # --- 3. Plotting the Curves ---
    plt.figure(figsize=(12, 7))
    
    # Using the same colors for consistency with the Standard plot
    colors = {
        'Tiny': '#1f77b4',   # Blue
        'Small': '#2ca02c',  # Green
        'Medium': '#ff7f0e', # Orange
        'Large': '#d62728',  # Red
        'XL': '#9467bd'      # Purple
    }

    # Plot each model's data
    for model, data in training_data.items():
        if not data['steps']:
            print(f"⚠️ Warning: No data found for {model}")
            continue
            
        plt.plot(data['steps'], data['losses'], label=f'{model} ($\mu$P)', 
                 color=colors.get(model, 'black'), linewidth=2, alpha=0.85)

    # --- 4. Chart Formatting ---
    plt.title('Training Loss vs. Steps ($\mu$P Models)', fontsize=16, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    
    # Add a grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add a legend
    plt.legend(title="Model Size", fontsize=11, title_fontsize=12)
    
    # --- 5. Save and Display ---
    output_filename = 'mup_training_curves.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\n🎉 Success! All {len(training_data)} model curves extracted and plotted.")
    print(f"Plot saved successfully as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()