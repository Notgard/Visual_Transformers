# %%
import math 
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device, device.type)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# %%
#emulating patch size of 4*4 using CNN 

height = 32
width = 32

color_channels = 3

patch_size = 4

# calculate the number of patches for the image
nb_patches = height * width // patch_size**2# 64 here

#use 2D conv layer generate patches
patchenizer = nn.Conv2d(in_channels=color_channels, out_channels=nb_patches, kernel_size=patch_size, stride=patch_size)
print(patchenizer)


# %%
#torch module that takes an image and returns a sequence of patches where the size of each patch is 4*4 and the output
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=64, img_size=32, nb_patches=0):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if nb_patches == 0:
            self.num_patches = (img_size // patch_size) ** 2
        else:
            self.nb_patches = nb_patches
        # Conv2d replaces manually extracting patches
        self.proj = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        #permute to have the batch size first
        x = x.permute(0, 2, 1) 
        #print(x.shape)
        return x

# %%
#take one image as tensor from train loader
image, _ = trainset[0]
print(image.shape)

patch_size = 4

color_channels = patch_size**2

# calculate the number of patches for the image
nb_patches = height * width // patch_size**2

#patch_embedding = PatchEmbedding(patch_size=patch_size, 
#                                 in_channels=color_channels, 
#                                 nb_patch=16)
#patches = patch_embedding(image)

patch_embed = PatchEmbedding(patch_size=4, in_channels=3, embed_dim=64, img_size=32, nb_patches=nb_patches)

image = image.unsqueeze(0)
print("Image Shape with Batch:", image.shape)

patches = patch_embed(image)

# %%
def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    #multiplication matricielle entre q et k
    attn_logits = q @ k.transpose(-2, -1)
    #scaling avec d_k (voir equation)
    attn_logits *= (1.0 / math.sqrt(d_k))
    #faire le softmax sur la dernière dimension
    attention = F.softmax(attn_logits, dim=-1)
    #multiplication matricielle avec v
    values = attention @ v
    return values

# %%
seq_len, d_k = 3, 2
q = torch.randn(seq_len, d_k)
k = torch.randn(seq_len, d_k)
v = torch.randn(seq_len, d_k)
values = scaled_dot_product(q, k, v)
print("Q\n", q.shape)
print("K\n", k.shape)
print("V\n", v.shape)
print("Values\n", values.shape)
#doit correspondre à la même shape

# %%
#do q k v projection with 3 nn.linear instead of just one mlp with 3 outputs (q, k, v heads)
class dumb_qkv_proj(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.q = nn.Linear(self.d_in, self.d_out)
        self.k = nn.Linear(self.d_in, self.d_out)
        self.v = nn.Linear(self.d_in, self.d_out)
        
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v

# %%
class qkv_proj(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Linear(self.d_in, self.d_out * 3)
    def forward(self, x):
        x = self.proj(x)
        return x.chunk(3, dim=-1)

# %%
proj = qkv_proj(d_in=2, d_out=2)
res = proj(values)
print(len(res))
q, k, v = res
print("Q\n", q.shape)
print("K\n", k.shape)
print("V\n", v.shape)

# %%
class qkv_proj(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_out = d_out

        self.proj = nn.Linear(self.d_in, self.d_out * 3)
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        #permute Batch, Head, SeqLen, Dims
        qkv = qkv.permute(0, 2, 1, 3) 
        return qkv.chunk(3, dim=-1)

# %%
q,k,v = qkv_proj(32, 32*3, 4)(torch.randn(1,64,32))
q.shape, k.shape, v.shape

# %%
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        #ps: juste pour vérifier que votre dimension
        #   match bien le nombre de tête...
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        #in_proj - votre fonction qkv_proj créé précédement
        self.in_proj = qkv_proj(input_dim, embed_dim, num_heads)
        
        self.o_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q, k, v = self.in_proj(x)
        #print(q.shape, k.shape, v.shape)
        #appeler votre fonction scaled_dot_product
        attn = scaled_dot_product(q, k, v)
        #print(attn.shape)
        #Permute back
        #Votre position de départ : [Batch, Head, SeqLen, Dims]
        #Permute dans la nouvelle dim : [Batch, SeqLen, Head, Dims]
        #(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = attn.permute(0, 2, 1, 3)
        #print(attn.shape)
        
        #reshape pour retirer la dimension "head".
        #aide : position de départ [Batch, SeqLen, Head, Dims]
        # position d'arrivée [Batch, SeqLen, self.embed_dim]
        attn = attn.reshape(batch_size, seq_length, self.embed_dim)
        #print(attn.shape)
        out = self.o_proj(attn)

        return out

# %%
net = MultiheadAttention(64, 64, 4)
net(torch.randn(1,16*16,64)).shape

# %%
net = MultiheadAttention(64, 256, 4)
net(torch.randn(1,16*16,64)).shape

# %%
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

# %%
class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super().__init__()

        # Multi-head Attention
        self.attention = MultiheadAttention(input_dim=input_dim, embed_dim=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)  # Normalisation après l'attention

        # Feed Forward
        self.ff = FeedForward(input_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)  # Normalisation après le feed forward

    def forward(self, x):
        # Multi-head Attention
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # Feed Forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

# %%
# Exemple d'entrée : séquence de patches (batch_size=1, num_patches=64, embed_dim=128)
x = torch.randn(1, 64, 128)

# Définition du Transformer Block
transformer = Transformer(input_dim=128, num_heads=8, hidden_dim=256)

# Passage avant
output = transformer(x)
print(output.shape)

print("image:", image.shape)
patches = patch_embed(image)
print("pacthes:", patches.shape)
transformer = Transformer(input_dim=64, num_heads=8, hidden_dim=256)
output = transformer(patches)
print(output.shape)

# %%
class TowerViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, hidden_dim, num_transformers, num_classes):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size=patch_size, 
                                              in_channels=in_channels, 
                                              nb_patches=((image_size*image_size) // patch_size) ** 2)

        num_patches = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        #stacking n number of transformers into a single sequential
        self.transformers = nn.Sequential(*[Transformer(embed_dim, num_heads, hidden_dim) for _ in range(num_transformers)])
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch Embedding
        x = self.patch_embedding(x)  # (batch, num_patches, embed_dim)
        
        # Ajout du token de classification
        cls_tokens = self.class_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Ajout de l'embedding positionnel
        x += self.pos_embedding

        # Passage à travers les blocs Transformer
        x = self.transformers(x)
        x = self.norm(x)

        # Classification (on récupère seulement le token [CLS])
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)


# %% [markdown]
# The ViT model was itself created as a combination of the specifications of the instructions for this project as well as insights I was able to gather from the paper which introduced vision transformers ("AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE", https://arxiv.org/abs/2010.11929).  
# The architecture in the paper which reflects parts of this implementation are as follows :
# ![vit_arch](vit_arch.png)

# %%
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, in_embed_dims, out_embed_dims, num_heads, hidden_dims, num_transformers, num_classes):
        super().__init__()

        assert len(in_embed_dims) == num_transformers, "Mismatch: in_embed_dims length must match num_transformers"
        assert len(out_embed_dims) == num_transformers, "Mismatch: out_embed_dims length must match num_transformers"
        assert len(hidden_dims) == num_transformers, "Mismatch: hidden_dims length must match num_transformers"
        
        num_patches = ((image_size*image_size) // patch_size) ** 2
        self.class_token = nn.Parameter(torch.randn(1, 1, in_embed_dims[0]))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, in_embed_dims[0]))

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, 
                                              in_channels=in_channels, 
                                              nb_patches=num_patches)

        # Transformer Blocks avec gestion de dimensions variables
        self.transformers = nn.ModuleList()
        for i in range(num_transformers):
            transformer = Transformer(in_embed_dims[i], num_heads[i], hidden_dims[i])
            self.transformers.append(transformer)

            if in_embed_dims[i] != out_embed_dims[i]:  # Changement de dimension si nécessaire
                self.transformers.append(nn.Linear(in_embed_dims[i], out_embed_dims[i]))

        # Dernière dimension de l'embed après tous les transformers
        last_embed_dim = out_embed_dims[-1]

        self.norm = nn.LayerNorm(last_embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(last_embed_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        batch_size, channels, width, height = x.shape

        # Patch Embedding
        x = self.patch_embedding(x)

        # Ajout du token de classification
        cls_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        #x += self.pos_embedding

        # Passage à travers les Transformers
        for layer in self.transformers:
            x = layer(x)

        x = self.norm(x)

        # Classification (on prend le token CLS)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

# %%
image_size = 32
patch_size = 4
in_channels = 3

embed_dim = 64
hidden_dim = 128

n_heads = 4
num_transformers = 3

num_classes = 10

towervit = TowerViT(image_size, patch_size, in_channels, embed_dim, n_heads, hidden_dim, num_transformers, num_classes)

x = torch.randn(1, 3, 32, 32)
output = towervit(x)
print(output.shape)

# %%
print(summary(towervit, (1, in_channels, image_size, image_size)))

# %%
BATCH_SIZE = 128
epochs = 200
learning_rate = 3e-4
towervit = towervit.to(device)

#using the same optimizer as in the ViT paper
opt = torch.optim.SGD(towervit.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, pin_memory=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/towerVIT_trainer_validation_{}'.format(timestamp))
training_and_validation_loop(towervit, train_loader, test_loader, epochs, writer, device, opt, timestamp, flatten=False)

# %%
image_size = 32
patch_size = 4
in_channels = 3

in_embed_dims = [64, 128, 256]  
out_embed_dims = [128, 256, 256]
num_heads = [4, 8, 16]
hidden_dims = [128, 256, 512]
num_transformers = 3
num_classes = 10

vit = ViT(image_size, patch_size, in_channels, in_embed_dims, out_embed_dims, num_heads, hidden_dims, num_transformers, num_classes)

x = torch.randn(1, 3, 32, 32)
output = vit(x)
print(output.shape)

# %%
print(summary(vit, (1, in_channels, image_size, image_size)))

# %%
BATCH_SIZE = 128
epochs = 200
learning_rate = 3e-4
vit = vit.to(device)

#using the same optimizer as in the ViT paper
opt = torch.optim.SGD(vit.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/VIT_trainer_validation_{}'.format(timestamp))
training_and_validation_loop(vit, train_loader, test_loader, epochs, writer, device, opt, timestamp, flatten=False)


