import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class ChartSummaryDataset(Dataset):
    """
    PyTorch Dataset for chart-to-text:
    Each sample returns (image_tensor, title_str, caption_str).
    Expects folder structure:
       root_dir/partX/imgs
       root_dir/partX/titles
       root_dir/partX/captions
    """
    def __init__(self, root_dir, part='part1', transform=None):
        super().__init__()
        self.part_dir = os.path.join(root_dir, part)
        self.img_dir = os.path.join(self.part_dir, 'imgs')
        self.title_dir = os.path.join(self.part_dir, 'titles')
        self.cap_dir = os.path.join(self.part_dir, 'captions')
        self.transform = transform

        img_extensions = ('*.png', '*.jpg', '*.jpeg')
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
        self.basenames = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        base = self.basenames[idx]
        # Load image
        imgs = [os.path.join(self.img_dir, base + ext) for ext in ('.png','.jpg','.jpeg')]
        img_path = next((p for p in imgs if os.path.exists(p)), None)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Load title
        title = ""
        for file in glob.glob(os.path.join(self.title_dir, base + '.*')):
            if file.endswith('.json'):
                import json
                data = json.load(open(file))
                title = data.get('title','') or str(data)
            else:
                title = open(file,'r').read().strip()
            break
        # Load caption
        caption = ""
        for file in glob.glob(os.path.join(self.cap_dir, base + '.*')):
            if file.endswith('.json'):
                import json
                data = json.load(open(file))
                caption = data.get('caption','') or str(data)
            else:
                caption = open(file,'r').read().strip()
            break
        return image, title, caption

def collate_fn(batch, vocab):
    images, _, captions = zip(*batch)
    images = torch.stack(images)
    sequences = [vocab.numericalize(c) for c in captions]
    max_len = max(len(seq) for seq in sequences)
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = torch.tensor(seq)
    return images, padded
