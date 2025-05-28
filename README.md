# Automated Chart-to-Text Summarization

An end-to-end PyTorch framework for generating natural-language summaries from chart images and metadata.

## Features

* **CNN + LSTM with Visual Attention**
  Extract spatial features from chart images (ResNet-18 or EfficientNet-B0 backbone) and decode into text via an LSTM decoder with additive attention over visual embeddings.

* **Staged Training Regime**

  1. Pure teacher-forcing warm-up
  2. Linear scheduled-sampling annealing
  3. Gradient clipping (max-norm = 1.0)
  4. Differential learning rates for backbone vs. decoder & title encoder

* **Beam-Search Inference**
  Configurable beam width for high-quality caption generation.

* **Comprehensive Evaluation**
  Token-level accuracy, BLEU, CIDEr, BLEURT, plus side-by-side visualizations and hooks for human evaluation.

## Repository Structure

```
chart-to-text/
├── data/
│   ├── pew_simple/part1/{imgs, titles, captions}
│   ├── pew_complex/…
│   ├── statista_simple/…
│   └── statista_complex/…
├── src/
│   ├── dataset.py       ← Dataset class & collate_fn
│   ├── vocab.py         ← Vocabulary builder
│   ├── model.py         ← CNNEncoder, Attention, LSTMDecoder
│   ├── train.py         ← Trainer with scheduled sampling
│   ├── inference.py     ← Beam-search & visualization
│   └── utils.py         ← Metrics, plotting, checkpointing
├── notebooks/           ← EDA and demo notebooks
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

## Installation

1. **Clone**
   `git clone https://github.com/your-username/chart-to-text.git`
   `cd chart-to-text`

2. **Create environment & install**
   `conda create -n chart2text python=3.9`
   `conda activate chart2text`
   `pip install -r requirements.txt`

3. **Verify GPU**

   ```python
   import torch
   print(torch.cuda.is_available() or torch.backends.mps.is_available())
   ```

## Data Preparation

Arrange your chart data as:

```
data/
  pew_simple/partX/{imgs, titles, captions}
  pew_complex/…
  statista_simple/…
  statista_complex/…
```

Point `--root_dir ./data` when running scripts, or edit `config.yaml` accordingly.

## Training

```bash
python src/train.py \
  --root_dir ./data \
  --backbone efficientnet_b0 \
  --embed_dim 192 \
  --hidden_dim 192 \
  --attn_dim 64 \
  --batch_size 12 \
  --lr_backbone 1e-4 \
  --lr_decoder 4e-4 \
  --warmup_epochs 3 \
  --ss_start 1.0 \
  --ss_end 0.1 \
  --clip_grad 1.0 \
  --epochs 20 \
  --device cuda
```

### Key arguments

* `--backbone`: `resnet18` or `efficientnet_b0` (default: `resnet18`)
* `--warmup_epochs`: epochs of pure teacher-forcing (default: 3)
* `--ss_start`, `--ss_end`: scheduled-sampling start/end probabilities (default: 1.0, 0.1)
* `--lr_backbone`, `--lr_decoder`: learning rates for backbone & head
* `--clip_grad`: gradient clipping max-norm (default: 1.0)

Checkpoints saved to `checkpoints/`.

## Inference

```bash
python src/inference.py \
  --checkpoint checkpoints/best.pt \
  --beam_width 4 \
  --num_samples 10 \
  --device cuda
```

Displays chart images with reference vs. predicted summaries.

## Evaluation Results

* **Token-Accuracy**: \~30% on held-out test
* **BLEU / CIDEr**: +25% improvement vs. baseline

## Contributing

1. Fork & clone
2. Install dependencies (`pip install -r requirements.txt`)
3. Implement your feature or fix
4. Submit a PR with tests or demo notebook

Please follow PEP8 and use Black formatting.


