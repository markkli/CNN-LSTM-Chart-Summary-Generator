import torch
import torch.nn as nn
import torchvision.models as models

class VisualAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()
        self.attn_feat = nn.Linear(feature_dim, attention_dim)
        self.attn_hidden = nn.Linear(hidden_dim, attention_dim)
        self.attn_v = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        feat_proj = self.attn_feat(features)
        hidden_proj = self.attn_hidden(hidden_state).unsqueeze(1)
        energy = torch.tanh(feat_proj + hidden_proj)
        scores = self.attn_v(energy).squeeze(2)
        alpha = torch.softmax(scores, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim, backbone='resnet18'):
        super().__init__()
        if backbone=='resnet18':
            base = models.resnet18(pretrained=True)
            channels = 512
        else:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            ef = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            base = ef
            channels = 1280
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.adapt = nn.Linear(channels, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images):
        x = self.backbone(images)
        B,C,H,W = x.size()
        x = x.view(B,C,-1).permute(0,2,1)
        x = self.adapt(x)
        return self.dropout(x)

class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, attention_dim):
        super().__init__()
        self.attention = VisualAttention(embed_dim, hidden_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim*2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features, inputs):
        embeddings = self.embedding(inputs)
        h = torch.zeros(1, embeddings.size(0), self.lstm.hidden_size, device=features.device)
        c = torch.zeros_like(h)
        outputs = []
        for t in range(embeddings.size(1)):
            context, _ = self.attention(features, h.squeeze(0))
            lstm_in = torch.cat([embeddings[:,t,:], context], dim=1).unsqueeze(1)
            out,(h,c) = self.lstm(lstm_in, (h,c))
            out = self.dropout(out.squeeze(1))
            outputs.append(self.fc(out))
        return torch.stack(outputs, dim=1)

class Chart2TextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, attention_dim=64, backbone='resnet18'):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim, backbone)
        self.decoder = LSTMDecoder(embed_dim, hidden_dim, vocab_size, attention_dim)

    def forward(self, images, captions):
        feats = self.encoder(images)
        return self.decoder(feats, captions)
