import os
import matplotlib.pyplot as plt

# Criar pasta de gráficos automaticamente
os.makedirs("graficos", exist_ok=True)

# Gráfico 1: Distribuição original das classes
classes_originais = ["Não tóxico", "Tóxico"]
quantidades_originais = [144277, 15294]

plt.figure(figsize=(7, 5))
plt.bar(classes_originais, quantidades_originais)
plt.title("Distribuição Original das Classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade de Comentários")
plt.savefig("graficos/grafico_distribuicao_original.png", dpi=300, bbox_inches="tight")
plt.close()

# Gráfico 2: Dataset balanceado
classes_balanceadas = ["Não tóxico", "Tóxico"]
quantidades_balanceadas = [3000, 3000]

plt.figure(figsize=(7, 5))
plt.bar(classes_balanceadas, quantidades_balanceadas)
plt.title("Distribuição Após Balanceamento")
plt.xlabel("Classe")
plt.ylabel("Quantidade de Comentários")
plt.savefig("graficos/grafico_distribuicao_balanceada.png", dpi=300, bbox_inches="tight")
plt.close()

# Gráfico 3: Loss durante o treinamento
etapas = [50, 100, 150, 200, 250, 300, 350, 400, 450]
loss = [0.5045, 0.2656, 0.2970, 0.2294, 0.3351, 0.2807, 0.2285, 0.2292, 0.2458]

plt.figure(figsize=(7, 5))
plt.plot(etapas, loss, marker="o")
plt.title("Evolução da Loss Durante o Treinamento")
plt.xlabel("Etapas de Treinamento")
plt.ylabel("Loss")
plt.savefig("graficos/grafico_loss_treinamento.png", dpi=300, bbox_inches="tight")
plt.close()

print("Gráficos gerados com sucesso!")
print("Arquivos salvos na pasta: graficos/")