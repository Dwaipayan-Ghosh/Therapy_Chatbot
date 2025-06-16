# 📌 1. INSTALL REQUIRED LIBRARIES
!pip install -q transformers datasets peft accelerate bitsandbytes trl einops

# 📌 2. MOUNT GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')
output_dir = "/content/drive/MyDrive/llama_therapy_finetuned"

# 📌 3. IMPORT LIBRARIES
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# 📌 4. LOAD DATASET FROM SPECIFIC PATH
file_path = "/content/mental_health_dataset (1).csv"
dataset = load_dataset("csv", data_files=file_path)["train"]

# 🔧 If your dataset has 4 columns (email, timestamp, input, response), rename them
if len(dataset.column_names) == 4:
    dataset = dataset.rename_columns({
        dataset.column_names[2]: "instruction",
        dataset.column_names[3]: "response"
    })

# 📌 5. FORMAT INTO PROMPT-RESPONSE STYLE
def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

dataset = dataset.map(format_prompt)

# 📌 6. LOAD TOKENIZER FROM SMALLER OPEN MODEL
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Open model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 📌 7. TOKENIZATION (smaller max_length to prevent crash)
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # 🔽 less memory
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 📌 8. LOAD MODEL (4-bit QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    bnb_4bit_compute_dtype=torch.float16
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# 📌 9. TRAINING CONFIG


from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# 📌 10. SELECT SMALL SET TO AVOID OOM
tiny_dataset = tokenized_dataset.select(range(min(100, len(tokenized_dataset))))

# 📌 11. TRAIN THE MODEL
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tiny_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# 📌 12. SAVE TO GOOGLE DRIVE
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model successfully saved to your Google Drive folder: {output_dir}")
