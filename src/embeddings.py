"""
embeddings.py

Batch computation of OpenCLIP, RAD-DINO, and BioMed-RoBERTa embeddings with caching.
"""
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext

from .config import configure_torch_for_gpu

class ImagePathDataset(Dataset):
    def __init__(self, filenames, images_dir):
        self.filenames = list(filenames)
        self.images_dir = Path(images_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = self.images_dir / filename
        image = Image.open(image_path).convert("RGB")
        return filename, image

def make_openclip_collate(preprocess):
    def collate(batch):
        names = []
        tensors = []
        for filename, image in batch:
            names.append(filename)
            tensors.append(preprocess(image))
        return names, torch.stack(tensors)
    return collate

def make_hf_vision_collate(processor):
    def collate(batch):
        names = []
        images = []
        for filename, image in batch:
            names.append(filename)
            images.append(image)
        return names, processor(images=images, return_tensors="pt")
    return collate

def compute_openclip_image_embeddings(filenames, images_dir, preprocess, model, device, batch_size=128, num_workers=2):
    configure_torch_for_gpu()
    dataset = ImagePathDataset(filenames, images_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        collate_fn=make_openclip_collate(preprocess),
    )

    filename_to_embedding = {}
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

    with torch.inference_mode():
        for batch_names, batch_tensors in tqdm(loader, desc="OpenCLIP image batches"):
            batch_tensors = batch_tensors.to(device, non_blocking=(device == "cuda"))
            with autocast_ctx:
                batch_embeddings = model.encode_image(batch_tensors)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
            embeddings_np = batch_embeddings.cpu().numpy()
            for fname, emb in zip(batch_names, embeddings_np):
                filename_to_embedding[fname] = emb

    return filename_to_embedding

def compute_hf_vision_embeddings(
    filenames, images_dir, processor, model, device, batch_size=96, num_workers=2, checkpoint_path=None, checkpoint_every=10
):
    configure_torch_for_gpu()
    dataset = ImagePathDataset(filenames, images_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        collate_fn=make_hf_vision_collate(processor),
    )

    filename_to_embedding = {}
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    with torch.inference_mode():
        for batch_idx, (batch_names, batch_inputs) in enumerate(tqdm(loader, desc="RAD-DINO batches"), start=1):
            batch_inputs = {
                key: value.to(device, non_blocking=(device == "cuda"))
                for key, value in batch_inputs.items()
            }
            with autocast_ctx:
                outputs = model(**batch_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
            embeddings_np = batch_embeddings.cpu().numpy()
            for fname, emb in zip(batch_names, embeddings_np):
                filename_to_embedding[fname] = emb

            if checkpoint_path is not None and batch_idx % checkpoint_every == 0:
                with checkpoint_path.open("wb") as f:
                    pickle.dump(filename_to_embedding, f)

    if checkpoint_path is not None:
        with checkpoint_path.open("wb") as f:
            pickle.dump(filename_to_embedding, f)

    return filename_to_embedding

def compute_text_embeddings(text_list, model, tokenizer, device, batch_size=128):
    configure_torch_for_gpu()
    embeddings = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

    with torch.inference_mode():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Text embedding batches"):
            inputs = tokenizer(
                text_list[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with autocast_ctx:
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

    return np.vstack(embeddings)
