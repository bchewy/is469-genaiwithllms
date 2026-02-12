"""
Quick BERT fine-tuning test on GPU - Sentiment Classification
Fine-tunes DistilBERT on a small subset of SST-2 (Stanford Sentiment Treebank)
"""
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def main():
    # --- GPU check ---
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load dataset (small subset for speed) ---
    print("\n--- Loading SST-2 dataset ---")
    dataset = load_dataset("glue", "sst2")
    
    # Use 2000 train, 500 eval for a quick test
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))
    eval_dataset = dataset["validation"].shuffle(seed=42).select(range(500))
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    
    # --- Tokenizer & Model ---
    model_name = "distilbert-base-uncased"
    print(f"\n--- Loading {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count / 1e6:.1f}M")
    
    # --- Tokenize ---
    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    
    # --- Training ---
    training_args = TrainingArguments(
        output_dir="./finetune_output",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        fp16=torch.cuda.is_available(),  # Mixed precision on GPU
        report_to="none",
        dataloader_num_workers=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\n--- Fine-tuning on {device.upper()} ---")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    
    # --- Eval ---
    results = trainer.evaluate()
    print(f"\n{'='*50}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Eval accuracy: {results['eval_accuracy']:.4f}")
    print(f"Eval loss: {results['eval_loss']:.4f}")
    
    # --- Quick inference test ---
    print(f"\n--- Inference test ---")
    test_sentences = [
        "This movie was absolutely fantastic!",
        "Terrible waste of time, do not watch.",
        "It was okay, nothing special.",
    ]
    
    model.eval()
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        conf = torch.softmax(logits, dim=-1).max().item()
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"  [{label} {conf:.2%}] {sent}")
    
    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
