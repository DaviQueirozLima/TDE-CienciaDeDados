# Detecção de Toxicidade com Transformers

Projeto desenvolvido para a disciplina de Ciência de Dados utilizando Deep Learning e Processamento de Linguagem Natural (PLN) para classificação de comentários tóxicos.

## Objetivo

O projeto tem como objetivo treinar uma rede neural profunda baseada em Transformers para identificar comentários tóxicos em textos online, auxiliando na moderação automática de conteúdo digital.

## Como Obter o Dataset

O dataset utilizado no projeto não foi incluído diretamente no repositório devido ao tamanho do arquivo.

Para executar o projeto corretamente:

1. Acesse o dataset no Kaggle:

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

2. Faça o download do arquivo `train.csv`.

3. Coloque o arquivo `train.csv` na pasta principal do projeto, no mesmo diretório do arquivo `main.py`.

Após isso, o projeto poderá ser executado normalmente.

## Tecnologias Utilizadas

* Python
* Pandas
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* DistilBERT

## Funcionalidades

* Pré-processamento de dados
* Balanceamento de classes
* Tokenização textual
* Fine-tuning de Transformer
* Classificação de toxicidade textual
* Avaliação do modelo

## Estrutura do Projeto

```bash
.
├── main.py
├── testar_modelo.py
├── graficos.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Como Executar

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Treinar modelo

```bash
python main.py
```

### Testar modelo treinado

```bash
python testar_modelo.py
```

### Gerar gráficos

```bash
python graficos.py
```

## Exemplos de Classificação

| Texto                         | Resultado  |
| ----------------------------- | ---------- |
| You are stupid and disgusting | Tóxico     |
| Thank you for helping me      | Não tóxico |

## Resultados

O modelo apresentou resultados satisfatórios na classificação de comentários tóxicos e não tóxicos, demonstrando a eficiência de Transformers em tarefas de Processamento de Linguagem Natural.

## Integrantes

* André Carlos Oliveira Da Costa
* Davi Queiroz Lima
* Gabriel Felipe Da Cruz Silva
* Kayo José Da Silva Fontes Lunga
* Kelson Kayque De Sousa Alves
* Lemuel Pinho De Oliveira
