# CS-GY-6923-SVG-Generator-Project
This project trains a suite of custom GPT-style autoregressive Transformers—ranging from a "Tiny" (128-dim) to an "XL" (768-dim) model—on a dataset of raw SVG code.

Install dependencies:Bashpip install -r requirements.txt
System Requirements (CairoSVG):Rendering SVGs to PNG files requires the system-level cairo library.
Ubuntu/Linux: sudo apt-get install libcairo2-devMacOS: brew install cairoWindows: Install GTK3 for Windows or disable .png saving in Part5_generating_samples.py.

🚀 Usage Guide / Pipeline1. Data Preparation
Run the data preparation script to clean the SVGs and generate svg_bpe_tokenizer.json and the split files in data_splits/.Bashpython prepare_svg_data.py

2. Training ModelsTo reproduce the scaling experiments, run the training scripts. You can train the Standard SP models or the muP models:Bashpython train_scaling.py
python mup_train_scaling.py

(Note: Training the Large and XL models requires a GPU with sufficient VRAM, such as an NVIDIA T4 or A100).

3. Visualizing ResultsGenerate the scaling law graphs and training curves by running the plotting scripts:
Bashpython Normal_ANDmup_scaling_plot.py
python predict_scaling.py

4. Generation & Inference
To generate new SVGs from the trained XL model checkpoint, run the generation script. This will output cleaned .svg files and rendered .png files to the output directories.
Bashpython Part5_generating_samples.py

📊 Key FindingsScaling Predictability: muP successfully generalized optimal learning rates across 5 model scales, allowing for highly accurate power-law loss extrapolation compared to Standard Parameterization.
Model Performance: The muP XL model achieved a highly confident validation perplexity of 1.9127.

Generative Capability: Despite the strict XML syntax requirements of SVGs, the autoregressive model successfully learned continuous geometric paths and spatial reasoning, accurately completing prefix-conditional prompts like connecting paths and lines.📄 ReferencesKaplan, J., et al. (2020). 
Scaling laws for neural language models. Yang, G., et al. (2022). Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer (muP.""")

