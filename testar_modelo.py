from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar modelo treinado
model_path = "./modelo_final"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = {
    0: "Não tóxico",
    1: "Tóxico"
}

def classificar_texto(texto):
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilidades = torch.softmax(outputs.logits, dim=1)
    classe = torch.argmax(probabilidades, dim=1).item()
    confianca = probabilidades[0][classe].item()

    return labels[classe], confianca

textos = [
    "You are a great person",
    "You are stupid and disgusting",
    "Thank you for helping me",
    "I hate you, you are horrible"
]

for texto in textos:
    classe, confianca = classificar_texto(texto)

    print(f"Texto: {texto}")
    print(f"Classificação: {classe}")
    print(f"Confiança: {confianca:.2%}")
    print("-" * 50)