import os
import re
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from glob import glob


def trim_to_77_tokens(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    result = []
    word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if word_count + sentence_word_count <= 77 / 1.4:
            result.append(sentence)
            word_count += sentence_word_count
        else:
            break

    return ' '.join(result)


class LaionDataset(Dataset):
    def __init__(self, image_root):
        self.image_root = image_root
        self.transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # SDXL expects [-1, 1]
        ])

        # Collect all existing .jpg paths in all shard folders
        image_paths = []
        shard_folders = sorted([f for f in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, f))])

        for shard in tqdm(shard_folders, desc="Scanning shard folders"):
            shard_path = os.path.join(image_root, shard)
            jpgs = sorted(glob(os.path.join(shard_path, "*.jpg")))
            image_paths.extend(jpgs)
        self.image_paths = image_paths

        print(f"Loaded {len(self.image_paths)} images from {len(shard_folders)} shards.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with open(image_path.replace('.jpg', '.txt'), 'r') as f:
            caption = f.readline().strip()

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = trim_to_77_tokens(caption)

        return caption, image
