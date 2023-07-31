# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:42:59 2023

@author: PauloAndrade
"""

#BIBLIOTECAS UTILIZADAS
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import openpyxl
import matplotlib.pyplot as plt
import nltk
import string
import re
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from nltk import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

# Na primeira executação é necessário a execução desta linha para baixar arquivos do NLTK necessário.
# nltk.download()


#Carrega a base inicial
base_inicial = pd.read_csv('base_tags.csv')

#Dividindo a base em 2 (1 - treino e 2 - Teste do modelo)

base_treino = base_inicial.head(int(len(base_inicial) * 0.7))  #Usaremos 70% da base para treino
base_teste = base_inicial.drop(base_treino.index) # Salvando os 30% restante que será utilizado ao final

#Padroniza o df
base_treino = base_treino.rename(columns={'tags':'label','post':'text'})
base_treino['all_old'] = base_treino['text'].copy()


#Quatidade de Registros
print(f'Qtd de Registro: {len(base_treino)}')

#Quantidade de Labels
print(f"Qtd de Categorias: {base_treino['label'].nunique()}")

#Distribuição dos valores entre os labels
groupby_label = base_treino.groupby('label')['text'].count().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='label', y='text', data=groupby_label)

# Adicione os rótulos de valor acima de cada barra
for index, value in enumerate(groupby_label['text']):
    plt.text(index, value, str(value), ha='center', va='bottom')

# Defina o título e os rótulos dos eixos
plt.title('Quantidade de registros por tag')
plt.xlabel('Tags')
plt.ylabel('Quantidade de Registros')

# Exiba o gráfico
plt.show()


#%Distribuição dos labels 
base_treino['label'].value_counts().plot(
    kind = 'pie', 
    explode = [0.1]*len(base_treino['label'].value_counts()) ,
    figsize = (5,5), autopct = '%1.1f%%',
    shadow = True)

plt.title ('%Distribuição de Categorias')
plt.show()



#Nuvem de Palavras Antes do processamento
wc = WordCloud(background_color =  'white')
wc.generate(str(base_treino['text']))
plt.imshow(wc, interpolation = 'bilinear')
plt.title('Nuvem Antes do Processamento')
plt.axis('off')
plt.show()

#Carrega os stopwords que serão removidos
stop_words = set(stopwords.words('portuguese'))
stop_words.add('digit') #Um termo que aparece na base e vamos retirar manualmente

# Inicializa o lematizador
lemmatizer = WordNetLemmatizer()



def pre_processamento(texto):
    # LOWER text
    texto = texto.lower()
    
    # PONTUAÇÃO --> remove todos as pontuações dos textos
    texto = re.sub(r'[^a-zA-Z]+', " ", texto)
    
    # TOKEN --> dividir frases em palavras ou tokens individuais
    tokens = word_tokenize(texto)
    
    # REMOVE STOPWORDS - remove as palavras mais comuns e sem significado do texto
    tokens = [word for word in tokens if word not in stop_words]
    
    # LEMMATIZA
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)


#Aplicando preprocessamento na coluna text
base_treino['text'] = base_treino['text'].apply(pre_processamento)

#Nuvem de Palavras com os stopwords
wc = WordCloud(background_color =  'white')
wc.generate(str(stopwords.words('english')))
plt.imshow(wc, interpolation = 'bilinear')
plt.title('Nuvem de stop words')
plt.axis('off')
plt.show()

#Nuvem de palavra após processamento
wc = WordCloud(background_color =  'white')
wc.generate(str(base_treino['text']))
plt.imshow(wc, interpolation = 'bilinear')
plt.title('Nuvem após Processamentos')
plt.axis('off')
plt.show()


#BAG OF WORDS - Transforma o texto em uma matriz de números
vectorizer = CountVectorizer()
vec_transform = vectorizer.fit_transform(base_treino['text'])  #treinamento
data_count_vector = np.array(vec_transform.todense(), dtype='float32')

#Definindo as variavéis de treinamento
x_train , x_test , y_train , y_test = train_test_split(
    data_count_vector , base_treino['label'], test_size = 0.2, random_state = 0)

# Carregando Modelo Naive Bayers
naive = MultinomialNB()
modelo = naive.fit(x_train,y_train)

# Rodando o modelo e buscando os resultados
predict_train = modelo.predict(x_train)
predict_test = modelo.predict(x_test)

print('Classification Reporte \n',metrics.classification_report(y_train, predict_train))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(y_train, predict_train))
print('\n')
print('Accuracy of train : {0:0.3f}'.format(metrics.accuracy_score(y_train,predict_train)))

print('Classification Reporte \n',metrics.classification_report(predict_test,y_test))
print('\n')
print('Confusion Matrix \n',metrics.confusion_matrix(predict_test,y_test))
print('\n')
print('Accuracy of test : {0:0.3f}'.format(metrics.accuracy_score(predict_test,y_test)))

heat_map = sns.heatmap(
    data = pd.DataFrame(confusion_matrix(y_train,predict_train)),
    annot = True,
    fmt = 'd',
    cmap = sns.color_palette('Blues',50),
    )



# Salvando o modelo em .pickle
with open ('modelo_vectorizer_naive_bayes.pkl' , 'wb') as file:
    pickle.dump(modelo, file)



# Atribuindo o predict a base e salvando com resultado
base_treino['predict'] = modelo.predict(data_count_vector)
base_treino.to_csv('base_predict.csv', index = False)


frase = 'jogou super bem'
frase_transform = vectorizer.transform([frase])
modelo.predict(frase_transform)