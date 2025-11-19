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
OUTPUT_DIR = "./phi3-cricket-finetuned-v2"
DATASET_PATH = "cricket_train_fixed.jsonl"

# LoRA configuration - INCREASED for better learning
lora_config = LoraConfig(
    r=32,  # Increased from 16
    lora_alpha=64,  # Increased from 32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def format_prompt(instruction, input_text, output_text=None):
    """
    FIXED: Format prompt correctly without 'nan'
    """
    # Combine instruction and input properly
    if input_text and input_text.strip() and input_text != "nan":
        full_question = f"{instruction}\n{input_text}"
    else:
        full_question = instruction
    
    if output_text:
        # Training format - include the answer
        return f"<|user|>\n{full_question}<|end|>\n<|assistant|>\n{output_text}<|end|>"
    else:
        # Inference format - no answer
        return f"<|user|>\n{full_question}<|end|>\n<|assistant|>\n"

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
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\n✗ No GPU detected! Training will be slow.")
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
    
    # CRITICAL: Show examples of formatted data
    print("\n" + "=" * 50)
    print("Sample Training Examples:")
    print("=" * 50)
    for i in range(min(2, len(dataset))):
        formatted = format_prompt(
            dataset[i]['instruction'],
            dataset[i]['input'],
            dataset[i]['output']
        )
        print(f"\nExample {i+1}:")
        print(formatted)
    print("=" * 50)
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments - INCREASED epochs and learning rate
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=20,  # INCREASED from 3 to 20
        learning_rate=5e-4,  # INCREASED from 2e-4
        fp16=True,
        logging_steps=2,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",  # Disable tensorboard
        max_grad_norm=0.3,
        logging_dir=None,
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
    print("Starting training...")
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total training steps: {len(tokenized_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    # Save the fine-tuned model
    print("\nSaving fine-tuned model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✓ Fine-tuning complete! Model saved to {OUTPUT_DIR}")
    print("\n" + "=" * 50)
    print("IMPORTANT: Update app.py to use new model path:")
    print(f"MODEL_PATH = './{OUTPUT_DIR}'")
    print("=" * 50)

if __name__ == "__main__":
    main()