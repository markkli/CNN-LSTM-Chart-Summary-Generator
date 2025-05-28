import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, ss_start=1.0, ss_end=0.1, warmup_epochs=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.ss_start = ss_start
        self.ss_end = ss_end
        self.warmup = warmup_epochs

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss=0; total_correct=0; total_tokens=0
        steps = len(self.train_loader)*(num_epochs - self.warmup)
        global_step = 0
        for images, captions in tqdm(self.train_loader):
            images,captions=images.to(self.device),captions.to(self.device)
            inputs=captions[:,:1]; h_c=None
            all_logits=[]
            for t in range(1,captions.size(1)):
                logits_t,h_c = self.model.decoder.step(self.model.encoder(images),
                                                        None,
                                                        inputs[:,-1:].to(self.device),
                                                        h_c)
                all_logits.append(logits_t)
                if epoch>self.warmup:
                    eps = max(self.ss_end, self.ss_start - global_step/steps*(self.ss_start-self.ss_end))
                    use_pred = (torch.rand(images.size(0),device=self.device) < (1-eps))
                    next_tok = logits_t.argmax(dim=-1,keepdim=True)
                    true_tok = captions[:,t].unsqueeze(1)
                    inp = torch.where(use_pred.unsqueeze(1), next_tok, true_tok)
                else:
                    inp = captions[:,t].unsqueeze(1)
                inputs = torch.cat([inputs,inp],dim=1)
                global_step +=1
            logits = torch.stack(all_logits,dim=1)
            targets = captions[:,1:]
            loss = self.criterion(logits.reshape(-1,logits.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            total_loss+=loss.item()
            preds=logits.argmax(dim=2)
            mask=targets.ne(self.criterion.ignore_index)
            total_correct+=(preds.eq(targets)&mask).sum().item()
            total_tokens+=mask.sum().item()
        return total_loss/len(self.train_loader), 100*total_correct/total_tokens

    def evaluate(self, loader):
        self.model.eval()
        total_correct=0; total_tokens=0
        with torch.no_grad():
            for images, captions in loader:
                images,captions=images.to(self.device),captions.to(self.device)
                logits=self.model(images,captions[:,:-1])
                preds=logits.argmax(dim=2)
                targets=captions[:,1:]
                mask=targets.ne(self.criterion.ignore_index)
                total_correct+=(preds.eq(targets)&mask).sum().item()
                total_tokens+=mask.sum().item()
        return 100*total_correct/total_tokens

    def train(self, num_epochs):
        start = time.time()
        for epoch in range(1, num_epochs+1):
            loss, acc = self.train_epoch(epoch, num_epochs)
            val_acc = self.evaluate(self.val_loader)
            print(f"Epoch {epoch}/{num_epochs} Loss={loss:.4f} TrainAcc={acc:.2f}% ValAcc={val_acc:.2f}%")
        print(f"Elapsed {time.time()-start:.1f}s")
