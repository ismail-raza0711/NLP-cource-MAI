# ============================================================
# train_poetry_model.py — GPT-2 Poem Generator (Clean + TF-IDF Edition)
# ============================================================

import os
import pandas as pd
import torch
from tqdm import tqdm
import spacy
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import Dataset

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


# ============================================================
# 0. Paths
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

paths = {
    "foundation": os.path.join(DATA_DIR, "PoetryFoundationData.csv"),
    "poetrydb": os.path.join(DATA_DIR, "PoetryDBData.csv"),
    "haiku": os.path.join(DATA_DIR, "HaikuDataset.csv"),
    "lyrics": os.path.join(DATA_DIR, "LyricsDataset.csv"),
}

print("📚 Loading datasets...")

dfs = []
for name, path in paths.items():
    print(f"📄 Loading {name}: {path}")
    df = pd.read_csv(path)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=["Poem"]).reset_index(drop=True)

print(f"📦 Total raw poems loaded: {len(df)}")


# ============================================================
# 1. Cleaning Functions
# ============================================================


def clean_poem(text):
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove invisible characters
    text = text.replace("\xa0", " ")

    # 3. Fix Haiku slash separators like "line / line / line"
    text = re.sub(r"\s*/\s*", "\n", text)

    # 4. Normalize punctuation
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")

    # 5. Remove bullet-style symbols
    text = re.sub(r"[•·►◇◆]+", "", text)

    # 6. Remove indentation
    text = "\n".join(line.strip() for line in text.splitlines())

    # 7. Collapse multiple blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # 8. Limit punctuation spam
    text = re.sub(r"[!?]{3,}", "!!", text)
    text = re.sub(r"[.]{3,}", "...", text)

    # 9. Collapse double spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ============================================================
# 2. Apply Cleaning with Progress Bars
# ============================================================

print("🧽 Cleaning poem texts...")
tqdm.pandas()

df["Poem"] = df["Poem"].astype(str).progress_apply(clean_poem)

# Clean titles too
if "Title" in df.columns:
    df["Title"] = df["Title"].astype(str).progress_apply(clean_poem)
else:
    df["Title"] = ""

df = df[df["Poem"].str.strip() != ""].reset_index(drop=True)
print(f"✅ Poems cleaned. Remaining: {len(df)}")


# ============================================================
# 3. Deduplicate
# ============================================================

print("🧹 Removing duplicates (Title + Poet)...")
initial_count = len(df)
df = df.drop_duplicates(subset=["Title", "Poet"], keep="first").reset_index(drop=True)
print(f"✔ Removed {initial_count - len(df)} duplicates.")


# ============================================================
# 4. TF–IDF Keyword Extraction
# ============================================================

print("📊 Building TF-IDF model...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=5000,
)

vectorizer.fit(df["Poem"].astype(str))
feature_names = vectorizer.get_feature_names_out()

print("✔ TF-IDF ready.")


# ============================================================
# 5. spaCy POS Filtering
# ============================================================

print("⚙️ Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

CUSTOM_BLACKLIST = {
    "love",
    "life",
    "time",
    "heart",
    "day",
    "night",
    "world",
    "dream",
    "light",
    "dark",
    "man",
    "woman",
    "eyes",
    "hand",
}


def extract_essence_keywords(poem_text, top_n=3):
    doc = nlp(poem_text.lower())

    allowed_tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.pos_ in {"NOUN", "ADJ", "VERB"}
        and not token.is_stop
        and token.lemma_ not in CUSTOM_BLACKLIST
    ]

    if not allowed_tokens:
        return []

    filtered_text = " ".join(allowed_tokens)
    tfidf_vec = vectorizer.transform([filtered_text]).toarray().flatten()

    ranked_words = []
    for idx in tfidf_vec.argsort()[::-1]:
        word = feature_names[idx]
        if word in allowed_tokens:
            ranked_words.append(word)
        if len(ranked_words) == top_n:
            break

    return ranked_words


# ============================================================
# 6. Build Prompt → Poem Training Pairs
# ============================================================


def make_prompt(row):
    poem_text = str(row["Poem"])

    title = row["Title"] if isinstance(row["Title"], str) else ""

    essence = extract_essence_keywords(poem_text)

    if essence:
        theme_words = " ".join(essence)
    elif title:
        theme_words = " ".join(title.split()[:3])
    else:
        theme_words = "memory shadow silence"

    return f"Write a poem about: {theme_words}\n{poem_text}"


print("📝 Creating training prompts...")
df["PromptedPoem"] = df.progress_apply(make_prompt, axis=1)
print("✔ PromptedPoem column created.")


# ============================================================
# 7. Preview 10 Samples AFTER TF-IDF + cleaning
# ============================================================

print("\n🔍 TEN RANDOM TRAINING EXAMPLES:\n")
samples = df.sample(10)
for _, row in samples.iterrows():
    print("-----------------------------------------------------")
    print(row["PromptedPoem"][:800], "...\n")


# ============================================================
# 8. Tokenization
# ============================================================

print("🔡 Loading tokenizer...")
dataset = Dataset.from_pandas(df[["PromptedPoem"]])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(
        examples["PromptedPoem"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


print("🔄 Tokenizing...")
tokenized_dataset = dataset.map(
    tokenize_function, batched=True, batch_size=32, desc="Tokenizing"
)

split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

print(
    f"✔ Tokenization complete. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
)


# ============================================================
# 9. Model Setup
# ============================================================

print("🧠 Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✔ Using: {device}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=os.path.join(ROOT_DIR, "poetry_model_output"),
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=2,
    save_total_limit=2,
    eval_strategy="epoch",  # <-- fixed
    save_strategy="epoch",
    logging_dir=os.path.join(ROOT_DIR, "logs"),
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


# ============================================================
# 10. Train
# ============================================================

print("🚀 Starting training...")
trainer.train()
print("🎉 Training complete!")


# Save model
model.save_pretrained(os.path.join(ROOT_DIR, "poetry_model"))
tokenizer.save_pretrained(os.path.join(ROOT_DIR, "poetry_model"))

print("💾 Model saved to poetry_model/")
