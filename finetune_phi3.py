import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./phi3-cricket-finetuned"
DATASET_PATH = "cricket_train_fixed.jsonl"

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def format_prompt(instruction, input_text, output_text=None):
    """Format the prompt for Phi-3"""
    if output_text:
        return f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n{output_text}<|end|>"
    else:
        return f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n"

def load_and_prepare_data():
    """Load JSONL dataset and prepare it"""
    dataset = load_dataset('json', data_files={'train': DATASET_PATH})
    return dataset['train']

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    prompts = []
    for i in range(len(examples['instruction'])):
        prompt = format_prompt(
            examples['instruction'][i],
            examples['input'][i],
            examples['output'][i]
        )
        prompts.append(prompt)
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def main():
    print("=" * 50)
    print("Starting Phi-3 Fine-tuning for Cricket Dataset")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\nâœ“ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"âœ“ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\nâœ— No GPU detected! Training will be slow.")
        return
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization to save VRAM
    print("\nLoading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=True
    )
    
    # Prepare model for LoRA
    print("\nPreparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load and prepare dataset
    print("\nLoading dataset...")
    dataset = load_and_prepare_data()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments optimized for RTX 3050 (4GB VRAM)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=5,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="tensorboard",
        max_grad_norm=0.3,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "=" * 50)
    print("Starting training... This may take 30-60 minutes!")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    # Save the fine-tuned model
    print("\nSaving fine-tuned model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nâœ“ Fine-tuning complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
