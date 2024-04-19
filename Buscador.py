# BUSCADOR
# O objetivo desse módulo é obter os resultados de um conjunto de buscas em um modelo salvo.

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
            if line.startswith('MODELO'):
                file = (line.split('=')[1])[1:-1]
                idf,tfidf,norm = read_json(file)

            elif line.startswith('CONSULTAS'):
                file = (line.split('=')[1])[1:-1]
                consultas = read_csv(file)

            elif line.startswith('RESULTADOS'):
                file = (line.split('=')[1])[1:-1]
                consulta(consultas,idf,tfidf,norm,file)

# Ler json de modelo
def read_json(json_file):
    logger.debug(f'Carregando modelo vetorial ({json_file})')
    with open(f'{json_file}', 'r') as f:
        modelo_json = json.load(f)

    return modelo_json['idf'],pd.DataFrame(modelo_json['tf_idf']),modelo_json['norm']

# Ler um arquivo csv
def read_csv(csv_file):
    logger.debug(f'Lendo arquivo CSV ({csv_file})')
    df = pd.read_csv(csv_file,sep=';', dtype={'QueryNumber': str})
    df['QueryText'] = df['QueryText'].apply(lambda x: eval(x))
    len_df = len(df)
    logger.debug(f'Total de consultas em {csv_file}: {len_df}')
    return df

# CONSULTA
def consulta(consultas,idf,tfidf,norm,file):
    results = []
    for query_number,query_text in zip(consultas['QueryNumber'], consultas['QueryText']):
        query_tfidf = tf_idf(query_text,idf)
        query_norm = euclidean_norm(query_tfidf)
        query_dot = dot(query_tfidf,tfidf)
        query_sim = similarity(query_dot,query_norm,norm)
        query_res = query_results(query_sim)

        logger.debug(f'Query {query_number}: (1) Documento {query_res[0][1]}, Similaridade {query_res[0][2]:.2f}')

        results.append((query_number,query_res))

    logger.debug(f'Salvando resultados em {file}.csv')
    df = pd.DataFrame(results, columns=['Query', 'Results'])
    df.to_csv(f'RESULT/{file}.csv', index=False, sep=';')
    
# TF-IDF (Term Frequency - Inverse Document Frequency)
# cada token na consulta tem peso 1, tfidf = idf
def tf_idf(token_list, idf):
    # logger.debug('Calculando TF-IDF (term frequency - inverse document frequency) da query')
    return pd.Series({t:v for t,v in idf.items() if t in token_list})

# Calcular a norma euclidiana
# |Di| = SQRT ( SUM (TF_IDF**2) )
def euclidean_norm(s):
    # logger.debug(f'Calculando norma euclidiana da query')
    return np.sqrt((s**2).sum())

# Calcular o produto escalar da query com documentos
def dot(query_tfidf,documents_tfidf):
    # logger.debug('Calculando os produtos escalares da query com documentos')
    return documents_tfidf.T[query_tfidf.index].multiply(query_tfidf, axis='columns').sum(axis=1)

# Calcular a similaridade da query com documento
def similarity(query_dot,query_norm,documents_norm):
    # logger.debug('Calculando a similaridade da query com documentos')
    similarities = query_dot.values.astype(float) / (query_norm * np.array(list(documents_norm.values())))
    return pd.Series(similarities, index=query_dot.index)

# Gerar lista com resultados de documentos mais próximos da query
def query_results(similarities):
    # logger.debug('Gerando os resultados da query')
    similarities = similarities.sort_values(ascending=False)

    query_results = []
    i = 0
    for doc,sim in similarities.items():
        i += 1
        query_results.append((i, doc, sim))
        
        # lista de resultados terá tamanho 10
        if(i == 10):
            break

    return query_results

def BUSCA():
    logger.debug('Iniciando BUSCADOR')
    ini = time.time()
    read_config('BUSCA.CFG')
    fim = time.time()
    logger.debug('Fim do BUSCADOR')
    logger.debug(f'Tempo de execução do BUSCADOR: {fim-ini:.2f}s')