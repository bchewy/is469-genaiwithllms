"""
BERT Fine-tuning on Yelp Review Full (5-class sentiment classification)
Manually tuned hyperparameters for best accuracy (no HPO).

Dataset: Yelp Review Full - 650K train, 50K test
Model: google-bert/bert-base-cased (110M params)
Task: 5-class classification (1-5 stars)
"""
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Paths (local cluster)
# ============================================================
DATASET_PATH = "/common/public/IS469/datasets/yelp_review_full/yelp_review_full/"
MODEL_PATH = (
    "/common/public/IS469/datasets/models--google-bert--bert-base-cased/"
    "snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e"
)

# ============================================================
# Hyperparameters (manually tuned — no HPO)
# ============================================================
MAX_LENGTH = 512            # Full BERT context; captures entire reviews
LEARNING_RATE = 2e-5        # Standard BERT fine-tuning sweet spot
NUM_EPOCHS = 3              # Sufficient for 650K samples
TRAIN_BATCH_SIZE = 8        # Per-device (conservative for 512 tokens)
EVAL_BATCH_SIZE = 32        # Larger for eval (no grad storage)
GRAD_ACCUM_STEPS = 4        # Effective batch = 8 * 4 = 32
WEIGHT_DECAY = 0.01         # L2 regularization
WARMUP_RATIO = 0.06         # ~6% of total steps
SEED = 42
EVAL_SUBSET_SIZE = 5000     # Intermediate eval subset (speed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


def main():
    print("=" * 60)
    print("BERT Fine-tuning on Yelp Review Full (5-class)")
    print("=" * 60)

    # --- GPU info ---
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {gpu_mem:.1f} GB")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load dataset ---
    print("\n--- Loading Yelp Review Full dataset ---")
    dataset = load_dataset(DATASET_PATH)
    print(f"Train: {len(dataset['train']):,} examples")
    print(f"Test:  {len(dataset['test']):,} examples")
    print(f"Classes: {dataset['train'].features['label'].names}")

    # Subset for intermediate eval (full test set used for final eval)
    eval_subset = dataset["test"].shuffle(seed=SEED).select(range(EVAL_SUBSET_SIZE))

    # --- Load model & tokenizer ---
    print(f"\n--- Loading BERT-base-cased ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

    # --- Tokenize (truncate only, dynamic padding via collator) ---
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    print(f"\n--- Tokenizing (max_length={MAX_LENGTH}, dynamic padding) ---")
    t0 = time.time()
    train_dataset = dataset["train"].map(tokenize_fn, batched=True, num_proc=4)
    eval_sub_tokenized = eval_subset.map(tokenize_fn, batched=True, num_proc=4)
    test_dataset = dataset["test"].map(tokenize_fn, batched=True, num_proc=4)
    print(f"Tokenization done in {time.time() - t0:.1f}s")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Print training config ---
    effective_batch = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
    total_steps = (len(train_dataset) // effective_batch) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"\n--- Training Configuration ---")
    print(f"Learning rate:         {LEARNING_RATE}")
    print(f"Epochs:                {NUM_EPOCHS}")
    print(f"Per-device batch:      {TRAIN_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUM_STEPS}")
    print(f"Effective batch size:  {effective_batch}")
    print(f"Weight decay:          {WEIGHT_DECAY}")
    print(f"Warmup ratio:          {WARMUP_RATIO} ({warmup_steps} steps)")
    print(f"Total steps (approx):  {total_steps}")
    print(f"Max sequence length:   {MAX_LENGTH}")
    print(f"FP16:                  {torch.cuda.is_available()}")
    print(f"Eval during training:  every 5000 steps ({EVAL_SUBSET_SIZE} samples)")
    print(f"Seed:                  {SEED}")

    training_args = TrainingArguments(
        output_dir="./bert_yelp_output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=2,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_sub_tokenized,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # --- Train ---
    print(f"\n{'=' * 60}")
    print(f"Starting fine-tuning on {device.upper()}...")
    print(f"{'=' * 60}\n")
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Training completed in {elapsed / 3600:.2f} hours ({elapsed:.0f}s)")
    print(f"Final training loss: {train_result.metrics['train_loss']:.4f}")

    # --- Final eval on FULL test set (50K) ---
    print(f"\n{'=' * 60}")
    print(f"Final Evaluation on full test set ({len(test_dataset):,} examples)")
    print(f"{'=' * 60}")

    test_results = trainer.evaluate(test_dataset)
    print(f"\n  Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"  Test Loss:     {test_results['eval_loss']:.4f}")

    # --- Detailed per-class metrics ---
    print(f"\n--- Generating detailed predictions on test set ---")
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    label_names = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

    print(f"\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=label_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    actual_pred = "Actual\\Pred"
    header = f"{actual_pred:>12}"
    for name in label_names:
        header += f"{name:>9}"
    print(header)
    for i, row in enumerate(cm):
        line = f"{label_names[i]:>12}"
        for val in row:
            line += f"{val:>9}"
        print(line)

    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i, name in enumerate(label_names):
        class_mask = labels == i
        class_acc = (preds[class_mask] == i).mean()
        print(f"  {name}: {class_acc:.4f} ({class_mask.sum():,} samples)")

    # --- Sample predictions ---
    print(f"\n--- Sample Predictions ---")
    sample_texts = [
        "This restaurant is absolutely amazing! Best food I've ever had. The service was impeccable and the atmosphere was perfect.",
        "Terrible service, cold food, rude staff. Never coming back. Complete waste of money.",
        "It was okay, nothing special but not bad either. Average food, average service.",
        "Pretty good experience overall. Food was tasty and the atmosphere was nice. Would recommend to friends.",
        "Decent place but overpriced. The portions were small for the price. Service was slow but friendly.",
    ]

    model.eval()
    for text in sample_texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1)[0]
        conf = probs.max().item()
        print(f"  [{label_names[pred]} ({conf:.1%})] {text[:80]}...")

    # --- GPU memory ---
    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # --- Save final model ---
    print(f"\n--- Saving final model to ./bert_yelp_final ---")
    trainer.save_model("./bert_yelp_final")
    tokenizer.save_pretrained("./bert_yelp_final")
    print("Model saved.")

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print(f"{'RESULTS SUMMARY':^60}")
    print(f"{'=' * 60}")
    print(f"  Model:              BERT-base-cased (110M params)")
    print(f"  Dataset:            Yelp Review Full (5-class)")
    print(f"  Train samples:      {len(train_dataset):,}")
    print(f"  Test samples:       {len(test_dataset):,}")
    print(f"  Max sequence length: {MAX_LENGTH}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  Epochs:             {NUM_EPOCHS}")
    print(f"  Effective batch:    {effective_batch}")
    print(f"  Training time:      {elapsed / 3600:.2f} hours")
    print(f"  Test Accuracy:      {test_results['eval_accuracy']:.4f}")
    print(f"  Test Loss:          {test_results['eval_loss']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
