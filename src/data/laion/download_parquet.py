from img2dataset import download
from datasets import load_dataset
import pandas as pd

# Load the metadata via streaming (no need to download the entire dataset)
dataset = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)

# Collect and filter samples
samples = []
for example in dataset:
    if example.get("URL") and example.get("similarity") and example.get("aesthetic"):
        samples.append({
            "URL": example["URL"],
            "TEXT": example.get("TEXT", ""),
            "similarity": example["similarity"],
            "aesthetic": example["aesthetic"]
        })
    if len(samples) >= 100_000:
        break

# Convert to DataFrame
df = pd.DataFrame(samples)

# Sort by aesthetic score + similarity
df_sorted = df.sort_values(by=["similarity", "aesthetic"], ascending=False)

# Take top 20k
df_top20k = df_sorted.head(20000)

# Save to CSV
df_top20k.to_csv("./src/data/laion/laion_top20k.csv", index=False)


download(
    url_list="./src/data/laion/laion_top20k.csv",
    input_format="csv",
    url_col="URL",
    caption_col="TEXT",
    output_format="files",
    output_folder="./src/data/laion/laion_top20k_images",
    processes_count=4,
    thread_count=16,
    resize_mode="no"
)
