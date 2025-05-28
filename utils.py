import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pytorch_metric_learning import losses

def compute_bleu(preds, refs):
    smooth = SmoothingFunction().method4
    return np.mean([sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
                    for p, r in zip(preds, refs)])

def compute_token_accuracy(preds, targets, pad_idx):
    correct = (preds == targets) & (targets != pad_idx)
    return correct.sum().item() / (targets != pad_idx).sum().item()

def save_checkpoint(model, optimizer, path, epoch):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict()
    }, path)
