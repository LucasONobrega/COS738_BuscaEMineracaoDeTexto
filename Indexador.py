# INDEXADOR
# A função desse módulo é criar o modelo vetorial, dadas as listas invertidas simples.

import pandas as pd
import numpy as np
import time
import json
from collections import defaultdict

import logging, logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('loggerBMT')

import xmltodict
from lxml import etree

# Ler um arquivo de configuração
def read_config(config_file):
    logger.debug(f'Lendo arquivo de configuração ({config_file})')
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('LEIA'):
                file = (line.split('=')[1])[1:-1]
                df = read_csv(file)

            elif line.startswith('ESCREVA'):
                file = (line.split('=')[1])[1:-1]
                index(df,file)
                
# Ler um arquivo csv
def read_csv(csv_file):
    logger.debug(f'Lendo arquivo CSV ({csv_file})')
    df = pd.read_csv(csv_file,sep=';',header=None)
    df.columns = ['TOKEN', 'LIST_DOCUMENTS']
    df['LIST_DOCUMENTS'] = df['LIST_DOCUMENTS'].apply(lambda x: eval(x))
    len_df = len(df)
    logger.debug(f'Total de tokens na lista invertida: {len_df}')
    return df

# Indexador
def index(df,file):
    logger.debug(f'Gerando indexador segundo modelo vetorial')
    dtm = document_term_matrix(df)
    tf = term_frequency(dtm)
    idf = inverse_document_frequency(dtm)
    tfidf = tf_idf(tf,idf)
    norm = euclidean_norm(tfidf)
    
    model = {
        'idf': idf.to_dict(),
        'tf_idf' : tfidf.to_dict(),
        'norm': norm.to_dict()
    }

    logger.debug(f'Salvando modelo vetorial em {file}.json')
    with open(f'RESULT/{file}.json', 'w') as f:
        json.dump(model, f)

# Gerar matriz termo documento
def document_term_matrix(df):
    logger.debug('Gerando a matriz termo documento')
    frequency = defaultdict(lambda: defaultdict(int))

    for token, list_documents in zip(df['TOKEN'], df['LIST_DOCUMENTS']):
        for doc in list_documents:
            frequency[doc][token] += 1

    df_frequency = pd.DataFrame(frequency).fillna(0)
    df_frequency = df_frequency.astype(int)

    df_frequency = df_frequency.sort_index()
    df_frequency = df_frequency.sort_index(axis=1)

    return df_frequency

# Calcular TF (Term Frequency)
# tf = freq / max freq
# freq = frequência do termo no documento
# max freq = frequência do termo que mais repete no documento
def term_frequency(df):
    logger.debug('Calculando TF (term frequency)')
    tf = df.apply(lambda x: x / x.max())
    return tf

# Calcular IDF (Inverse Document Frequency)
# idf = log (N / n)
# N = número total de documentos no sistema
# n = número de documentos que o termo aparece
def inverse_document_frequency(df):
    logger.debug('Calculando IDF (inverse document frequency)')
    N = len(df.columns)
    idf = np.log(N/df.astype(bool).sum(axis=1))
    return idf

# TF-IDF (Term Frequency - Inverse Document Frequency)
def tf_idf(tf,idf):
    logger.debug('Calculando TF-IDF (term frequency - inverse document frequency)')
    return tf.mul(idf, axis=0)

# Calcular a norma euclidiana
# |Di| = SQRT ( SUM (TF_IDF**2) )
def euclidean_norm(df):
    logger.debug(f'Calculando norma euclidiana dos documentos')
    return np.sqrt((df**2).sum())

def INDEX():
    logger.debug('Iniciando INDEXADOR')
    ini = time.time()
    read_config('INDEX.CFG')
    fim = time.time()
    logger.debug('Fim do INDEXADOR')
    logger.debug(f'Tempo de execução do INDEXADOR: {fim-ini:.2f}s')