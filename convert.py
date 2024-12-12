from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = SentenceTransformer(
    "all-MiniLM-L6-v2", device=device
)

df = pd.read_json("./startups_demo.json", lines=True)

vectors = model.encode(
    [row.alt + ". " + row.description for row in df.itertuples()],
    show_progress_bar=True,
)

np.save("startup_vectors.npy", vectors, allow_pickle=False)
