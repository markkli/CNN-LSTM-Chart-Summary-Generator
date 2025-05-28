import torch
import matplotlib.pyplot as plt
import textwrap
from tqdm import tqdm

def decode_sequence(seq, vocab):
    tokens=[]
    for idx in seq:
        if idx==vocab.stoi["<eos>"]: break
        if idx in (vocab.stoi["<pad>"],vocab.stoi["<sos>"]): continue
        tokens.append(vocab.itos.get(idx,"<unk>"))
    return " ".join(tokens)

def unnormalize(img_tensor, mean, std):
    return (img_tensor*std+mean).permute(1,2,0).cpu().numpy()

def generate_beam(model, images, vocab, device, beam_width=4, max_len=30):
    # ... copy generate_beam from notebook or utils ...
    pass

def visualize_batch(model, data_loader, vocab, device, beam_width=4):
    mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    model.eval()
    with torch.no_grad():
        for images,captions in data_loader:
            images,captions=images.to(device),captions.to(device)
            preds=generate_beam(model,images,vocab,device,beam_width, captions.size(1)-1)
            for i in range(min(5,images.size(0))):
                ref=decode_sequence(captions[i,1:].cpu().tolist(),vocab)
                pred=decode_sequence(preds[i],vocab)
                fig,axes=plt.subplots(2,1,figsize=(6,6))
                axes[0].imshow(unnormalize(images[i].cpu(),mean,std)); axes[0].axis('off')
                text=textwrap.fill(f"Ref: {ref}\nPred: {pred}",width=80)
                axes[1].text(0,0.5,text,fontsize=9)
                axes[1].axis('off')
                plt.tight_layout(); plt.show()
            break
