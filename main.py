import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1. Carregar dataset
df = pd.read_csv("train.csv")

# 2. Selecionar colunas
df = df[["comment_text", "toxic"]].dropna()

df = df.rename(columns={
    "comment_text": "text",
    "toxic": "label"
})

# 3. Balancear dataset
df_nao_toxico = df[df["label"] == 0].sample(3000, random_state=42)
df_toxico = df[df["label"] == 1].sample(3000, random_state=42)

df_balanceado = pd.concat([df_nao_toxico, df_toxico]).sample(frac=1, random_state=42)

print("Quantidade por classe:")
print(df_balanceado["label"].value_counts())

# 4. Separar treino e teste
train_df, test_df = train_test_split(
    df_balanceado,
    test_size=0.2,
    random_state=42,
    stratify=df_balanceado["label"]
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 5. Tokenização
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
test_dataset = test_dataset.remove_columns(["text", "__index_level_0__"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 6. Modelo Transformer
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 7. Configuração do treinamento
training_args = TrainingArguments(
    output_dir="./modelo_toxicidade",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 9. Treinar modelo
trainer.train()

# 10. Avaliar
resultado = trainer.evaluate()
print(resultado)

# 11. Salvar modelo final
trainer.save_model("./modelo_final")
tokenizer.save_pretrained("./modelo_final")

print("Modelo treinado e salvo com sucesso!")