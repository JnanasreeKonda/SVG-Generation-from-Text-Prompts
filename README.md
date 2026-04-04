# SVG-Generation-from-Text-Prompts

Fine-tuning a small language model to generate SVG code from natural language descriptions — Deep Learning Kaggle Competition submission.

## Authors

- [Jnanasree Konda] — [jk9286@nyu.edu]
- [Rithwik Amajala] — [sa9880@nyu.edu]

## What We're Doing

We fine-tune **Qwen2.5-1.5B-Instruct** using **QLoRA** (4-bit quantization + LoRA adapters) to take a text prompt like *"a red circle on a white background"* and output valid SVG code that draws it.

The model is trained on 12,000 cleaned examples from the competition's 50,000 (prompt, SVG) pairs. At inference, we use greedy decoding with a two-pass retry and XML repair to ensure all outputs meet the competition's SVG validity constraints.

## How to Run

### On Kaggle (recommended)

1. Import the notebook into a Kaggle session and enable a **GPU accelerator**.
2. Attach the competition dataset so `train.csv` and `test.csv` are available.
3. Click **Run All** — the notebook runs end-to-end in ~115 minutes on H100.
4. Download `submission.csv` from the output files panel.

### On Google Colab

1. Upload the notebook and set the runtime to **GPU**.
2. Upload `train.csv` and `test.csv` to the session storage.
3. Run all cells in order. `submission.csv` will download automatically.

### Dependencies

All packages are installed in **Cell 0** of the notebook:

```bash
pip install transformers peft accelerate bitsandbytes datasets scikit-learn lxml cairosvg
```

## Code

[Link your GitHub or Kaggle notebook URL here]

