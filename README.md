# SPAM Classification with NLP and NaiveBayes Models

## Descrição
Este notebook aborda a classificação de mensagens de texto em spam e ham utilizando técnicas de Processamento de Linguagem Natural (NLP) e um modelo de Naive Bayes. O fluxo de trabalho inclui desde o carregamento e exploração dos dados até o pré-processamento do texto, treinamento do modelo e avaliação dos resultados.

## Bibliotecas Utilizadas
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Operações numéricas
- **Matplotlib**: Visualização de dados
- **Seaborn**: Visualização de dados com gráficos aprimorados
- **Scikit-learn**: Modelagem e avaliação de dados
  - `train_test_split`
  - `CountVectorizer`
  - `TfidfTransformer`
  - `Pipeline`
  - `metrics`
  - `MultinomialNB`
- **WordCloud**: Geração de nuvem de palavras
- **NLTK**: Processamento de linguagem natural
  - `stopwords`
  - `pos_tag`
  - `word_tokenize`
  - `SnowballStemmer`
  - `WordNetLemmatizer`

## Dataset
O dataset utilizado é o `spam.csv`, que contém duas colunas principais:
- `class`: Indica se a mensagem é spam ou ham.
- `text`: A mensagem de texto a ser classificada.

### Resultados
Os resultados obtidos incluem a acurácia do modelo, a curva ROC que mostra a taxa de verdadeiros positivos versus a taxa de falsos positivos e a matriz de confusão que detalha o desempenho do modelo em termos de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
O ROC_Curver apresentou o resultado de 98% enquanto a Acurácia, apresentou 96%. Lembramos que é um dataset desbalanceado, portanto a acurácia não é uma boa métrica de avaliação.


### Conclusão
Este notebook demonstra como aplicar técnicas de NLP e modelos de classificação para detectar mensagens de spam. O uso de pipelines facilita a integração de várias etapas de pré-processamento e modelagem, resultando em um fluxo de trabalho eficiente e replicável.
