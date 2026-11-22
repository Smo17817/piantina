import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image, ImageOps
import pandas as pd
import random
import matplotlib.pyplot as plt
from transformers import AutoModel
import torch.nn.functional as F
import numpy as np

# --- Routes ---
BASE = Path(__file__).resolve().parent
ROOT = BASE / "dataset" / "house_plant_species"
SPLIT_CSV = BASE / "split" / "extended_split.csv"
OUT = BASE / "outputs"
EMBEDDING = OUT / "embeddings_extended.npz"

# --- Costanti --- 
MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 16
NUM_WORKERS = 0
RESIZE_SIZE = 256
IMAGE_SIZE = 224
# Costanti utilizzate su Dataset Imagenet per i canali RGB
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def build_transform():
    trasformazioni = [
        v2.Resize(RESIZE_SIZE), # Porta il lato corto a 256 pixel e scala l’altro in proporzione
        v2.CenterCrop(IMAGE_SIZE), # Ritaglia usando il formato scelto
        v2.ToImage(), # converte le immagini PIL in un tensore
        v2.ToDtype(torch.float32, scale=True), # converte in decimali, scale porta i valori tra 0 e 1
        v2.Normalize(mean=MEAN, std=STD) # Normalizzazione: applica (x-media)/ dev_std
    ]

    return v2.Compose(trasformazioni)

def check_tensor(x):
    print(x.shape)
    print(x.dtype)
    print(x.min())
    print(x.max())

def denormalize(x): 
    ''' Annulla la normalizzazione '''
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)

def show_samples(ds, n_rows=2, n_cols=3, seed=None):
    rng = random.Random(seed)
    indexes = rng.sample(range(len(ds)),  n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for ax, idx in zip(axes.flat, indexes):
        x, i = ds[idx]
        img = denormalize(x).permute(1, 2, 0).numpy()   # (3, H, W) → (H, W, 3)
        ax.imshow(img)
        ax.set_title(f"{ds.df.iloc[i]['specie']}\n(indice {i})", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    plt.show()

class PlantDataset(Dataset):
    def __init__(self, csv_path, root, transform):
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self.root / row["path"]

        with Image.open(img_path) as img:
            if img.mode == "P": 
                # Risolve un problema di Pillow per le immagini con formato P
                img = ImageOps.exif_transpose(img).convert("RGBA")
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")

        return (self.transform(rgb_img), i)

def load_model(device):
    ''' Caricamento del modello '''
    model = AutoModel.from_pretrained(MODEL_NAME) # Carichiamo il modello da HuggingFace
    model.eval() # Lo poniamo in modalità inferenza
    model.to(device) # Lo spostiamo sulla GPU se presente, altrimenti CPU

    return model

def extract(model, loader, device):
    batch_emb = []
    batch_idx = []
    
    with torch.inference_mode():    
        for n, (xb, ib) in enumerate(loader):
            outputs = model(pixel_values=xb.to(device))
            cls = outputs.last_hidden_state[:, 0]
            cls = F.normalize(cls, dim=1)
            batch_emb.append(cls.cpu())
            batch_idx.append(ib)
    
            if (n + 1) % 20 == 0 or n + 1 == len(loader):
                print(f"Batch: {n+1:>5}/{len(loader)}") 
    
    emb = torch.cat(batch_emb)
    idx = torch.cat(batch_idx)

    # Controllo rumoroso sulla corrispondenza tra gli indici degli embeddings e quelli originali
    assert torch.equal(idx, torch.arange(len(loader.dataset))), "L'ordine degli indici non corrisponde"

    return emb.numpy()

def save(embeddings, df, out_file):
    # Crea il genitore se non è presente
    out_file.parent.mkdir(exist_ok=True)

    paths = df["path"].to_numpy(dtype=str)
    specie = df["specie"].to_numpy(dtype=str)
    groups = df["group"].to_numpy(dtype=str)
    np.savez(out_file, embedding=embeddings, path=paths, specie=specie, group=groups)

def check(out_file):
    dati = np.load(out_file)
    emb = dati["embedding"]
    specie = dati["specie"]

    print(f"Controllo forma: {emb.shape}")
    print(f"Controllo specie: {len(specie)}")
    print(f"Controllo dimensione delle righe: {np.allclose(np.linalg.norm(emb, axis=1), 1)}")

    # --- Similarità ---

    # Embeddings
    # Viene calcolata la similarità come prodotto scalare tra gli embeddings (avendo lunghezza 1)
    # Più si somigliano, più S[i, j] ~ 1
    S = emb @ emb.T

    # Specie
    # Matrice T/F per veder quante coppie sono della stessa specie
    same = specie[:, None] == specie[None, :] # broadcasting tra la versione colonna e la versione riga di specie, che produce la matrice
    # Si elimina la diagonale perchè influenza la media
    diag = np.eye(len(emb), dtype=bool) # True solo sulla diagonale

    print(f"Similarità media tra immagini della stessa specie: {S[same & ~diag].mean()}")
    print(f"Similarità media tra immagini di specie diverse: {S[~same].mean()}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PlantDataset(SPLIT_CSV, ROOT, build_transform())

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = load_model(device)
    embedding = extract(model, loader, device)

    save(embedding, ds.df, EMBEDDING)

    check(EMBEDDING)