from transformers import Trainer, TrainingArguments
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = np.mean(preds == labels)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./models/wav2vec2-ser",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    gradient_checkpointing=True,
    report_to="none"
)

from transformers import DataCollatorCTCWithPadding

data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(100)),
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./models/wav2vec2-ser")  # ✅ explicitly save model here
processor.save_pretrained("./models/wav2vec2-ser")  # ✅ save processor too
