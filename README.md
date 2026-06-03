# Detecção de Toxicidade com Transformers

Projeto desenvolvido para a disciplina de **Fundamentos de Ciência de Dados** utilizando técnicas de **Deep Learning** e **Processamento de Linguagem Natural (PLN)** para classificação automática de comentários tóxicos.

---

# Objetivo

O objetivo deste projeto é treinar uma rede neural profunda baseada em **Transformers (DistilBERT)** para identificar comentários tóxicos em textos online, auxiliando na moderação automática de conteúdo digital.

A solução foi desenvolvida utilizando uma base de dados pública do Kaggle e técnicas modernas de Processamento de Linguagem Natural, permitindo a classificação automática de comentários em duas categorias:

* Não tóxico
* Tóxico

---

# Dataset Utilizado

O projeto utiliza o dataset público:

**Jigsaw Toxic Comment Classification Challenge**

Disponível em:

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

O dataset contém milhares de comentários classificados quanto à sua toxicidade, sendo amplamente utilizado em pesquisas relacionadas à moderação automática de conteúdo.

---

# Tecnologias Utilizadas

* Python
* Pandas
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Datasets
* Matplotlib

---

# Funcionalidades

* Carregamento e preparação dos dados
* Remoção de valores nulos
* Balanceamento das classes
* Separação entre treino e teste
* Tokenização textual
* Fine-tuning de Transformers
* Classificação de toxicidade textual
* Avaliação do modelo
* Geração automática de gráficos

---

# Estrutura do Repositório (GitHub)

Ao clonar o projeto, a estrutura será semelhante a:

```bash
.
├── graficos/
│   ├── grafico_distribuicao_original.png
│   ├── grafico_distribuicao_balanceada.png
│   └── grafico_loss_treinamento.png
│
├── main.py
├── testar_modelo.py
├── graficos.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Estrutura Após o Treinamento

Após executar:

```bash
python main.py
```

serão criadas automaticamente as seguintes pastas:

```bash
.
├── graficos/
│   ├── grafico_distribuicao_original.png
│   ├── grafico_distribuicao_balanceada.png
│   └── grafico_loss_treinamento.png
│
├── modelo_final/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
│
├── modelo_toxicidade/
│   ├── checkpoint-*
│   └── ...
│
├── train.csv
├── main.py
├── testar_modelo.py
├── graficos.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Observação

As pastas:

```txt
modelo_final/
modelo_toxicidade/
```

são geradas automaticamente durante o treinamento e não são armazenadas no GitHub devido ao tamanho dos arquivos gerados.

---

# Como Obter o Dataset

O arquivo `train.csv` não foi incluído no repositório devido ao tamanho do dataset.

## Passo 1

Acesse:

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

## Passo 2

Faça o download do arquivo:

```txt
train.csv
```

## Passo 3

Posicione o arquivo na raiz do projeto:

```bash
.
├── train.csv
├── main.py
├── testar_modelo.py
└── ...
```

---

# Instalação

Clone o repositório:

```bash
git clone https://github.com/DaviQueirozLima/TDE-CienciaDeDados.git
```

Entre na pasta do projeto:

```bash
cd TDE-CienciaDeDados
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

# Dependências

Arquivo `requirements.txt`:

```txt
pandas
scikit-learn
datasets
transformers
torch
accelerate
matplotlib
```

---

# Treinamento do Modelo

Após instalar as dependências e adicionar o dataset:

```bash
python main.py
```

Durante a execução serão realizadas as seguintes etapas:

1. Carregamento do dataset.
2. Seleção das colunas necessárias.
3. Balanceamento das classes.
4. Divisão entre treino e teste.
5. Tokenização dos textos.
6. Configuração do modelo DistilBERT.
7. Fine-tuning do Transformer.
8. Avaliação do desempenho.
9. Salvamento do modelo treinado.

Ao final da execução será exibida uma mensagem semelhante a:

```txt
Modelo treinado e salvo com sucesso!
```

Além disso, serão criadas automaticamente as pastas:

```txt
modelo_final/
modelo_toxicidade/
```

---

# Testando o Modelo Treinado

Após concluir o treinamento:

```bash
python testar_modelo.py
```

O sistema carregará automaticamente o modelo salvo na pasta:

```txt
modelo_final/
```

e realizará classificações em novos comentários.

---

# Exemplo de Saída

```txt
Texto: You are a great person
Classificação: Não tóxico
Confiança: 87.60%

Texto: You are stupid and disgusting
Classificação: Tóxico
Confiança: 99.28%

Texto: Thank you for helping me
Classificação: Não tóxico
Confiança: 99.14%

Texto: I hate you, you are horrible
Classificação: Tóxico
Confiança: 98.97%
```

---

# Gráficos

Os gráficos utilizados no relatório já estão disponíveis na pasta:

```txt
graficos/
```

Arquivos disponíveis:

```txt
grafico_distribuicao_original.png
grafico_distribuicao_balanceada.png
grafico_loss_treinamento.png
```

Caso seja necessário recriá-los:

```bash
python graficos.py
```

O script criará automaticamente a pasta:

```txt
graficos/
```

caso ela não exista.

---

# Resultados Obtidos

O modelo foi capaz de identificar corretamente comentários tóxicos e não tóxicos utilizando a arquitetura DistilBERT.

Os testes realizados demonstraram altos níveis de confiança nas classificações, evidenciando a eficiência dos Transformers em tarefas de Processamento de Linguagem Natural.

Além disso, os gráficos gerados permitiram visualizar:

* O desbalanceamento original das classes.
* O balanceamento realizado durante o pré-processamento.
* A evolução da função de perda (loss) durante o treinamento.

---

# Repositório

GitHub:

https://github.com/DaviQueirozLima/TDE-CienciaDeDados

---

# Integrantes

* André Carlos Oliveira Da Costa
* Davi Queiroz Lima
* Gabriel Felipe Da Cruz Silva
* Kayo José Da Silva Fontes Lunga
* Kelson Kayque De Sousa Alves
* Lemuel Pinho De Oliveira

---

# Licença

Projeto desenvolvido exclusivamente para fins acadêmicos na disciplina de Fundamentos de Ciência de Dados.
