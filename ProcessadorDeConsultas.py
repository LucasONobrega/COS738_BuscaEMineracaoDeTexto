# PROCESSADOR DE CONSULTAS
# O objetivo desse módulo é transformar o arquivo de consultas fornecido ao padrão de palavras que estamos utilizando.

import pandas as pd
import time
import re

import logging, logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('loggerBMT')

import xmltodict
from lxml import etree

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

bool_stemmer = False

# Ler um arquivo de configuração
def read_config(config_file):
    global bool_stemmer
    logger.debug(f'Lendo arquivo de configuração ({config_file})')
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('STEMMER'):
                bool_stemmer = True
            
            elif line.startswith('LEIA'):
                file = (line.split('=')[1])[1:-1]
                xml = read_xml(file)

            elif line.startswith('CONSULTAS'):
                file = (line.split('=')[1])[1:-1]
                csv_consultas(xml,file)

            elif line.startswith('ESPERADOS'):
                file = (line.split('=')[1])[1:-1]
                csv_esperados(xml,file)

# Ler um arquivo em formato XML
def read_xml(xml_file):
    logger.debug(f'Lendo arquivo XML ({xml_file})')
    parser = etree.XMLParser(dtd_validation=True)
    tree = etree.parse(f'{xml_file}', parser)
    dict = xmltodict.parse(etree.tostring(tree))
    len_dict = len(dict['FILEQUERY']['QUERY'])
    logger.debug(f'Total de consultas em {xml_file}: {len_dict}')
    
    return dict['FILEQUERY']['QUERY']
    # list = [ { 'QueryNumber' : STRING
    #            'QueryText'   : STRING
    #            'Results'     : STRING
    #            'Records'     : { 'Item' : [ { '@score' : STRING
    #                                           '#text'  : STRING }, ... ]
    #                            } 
    #          }, ... 
    #        ]

# Remover stopwords
def remove_stop_words(tokens): 
    stop_words = set(stopwords.words('english'))
    stop_words.add('patient')
    stop_words.add('patients')
    ft = [t for t in tokens if t.lower() not in stop_words] 
    return ft

# Remover pontuação
def remove_punct(string):
    return re.sub(r'[^\w\s]', ' ', string)

# Remover tamanho
def remove_len(tokens):
    ft = [t for t in tokens if len(t)>=5] 
    return ft

# Porter Stemming
def stemming(tokens):
    st = [PorterStemmer().stem(t) for t in tokens]
    return st

# Upper
def upper(tokens):
    u = [str.upper(t) for t in tokens]
    return u

# Gerar arquivo indicado na instrução CONSULTAS
def csv_consultas(xml,file):
    logger.debug(f'Gerando arquivo da instrução CONSULTAS')

    df = pd.DataFrame(xml)
    df = df[['QueryNumber', 'QueryText']]

    # remover pontuação, tokenização, lematização, remover stopwords, remover tokens de tamanho menor que 5, maiúscula
    df['QueryText'] = df['QueryText'].apply(remove_punct)
    df['QueryText'] = df['QueryText'].apply(word_tokenize)
    df['QueryText'] = df['QueryText'].apply(remove_stop_words)
    df['QueryText'] = df['QueryText'].apply(remove_len)
    
    if(bool_stemmer == True):
        file += '-STEMMER'
        df['QueryText'] = df['QueryText'].apply(stemming) 
    else:
        file += '-NOSTEMMER'

    df['QueryText'] = df['QueryText'].apply(upper)

    logger.debug(f'Salvando arquivo {file}.csv')
    df.to_csv(f'RESULT/{file}.csv', index=False, sep=';')

# Gerar arquivo indicado na instrução ESPERADOS
def csv_esperados(xml,file):
    logger.debug(f'Gerando arquivo da instrução ESPERADOS')

    df = pd.DataFrame(xml)
    df = df[['QueryNumber', 'Records']]

    df['Records'] = df['Records'].apply(lambda x: x['Item'])
    df = df.explode('Records')
    df['DocVotes'] = df['Records'].apply(lambda x: sum(int(i) for i in x['@score'][:4]))
    df['DocNumber'] = df['Records'].apply(lambda x: x['#text'])
    df.drop(columns=['Records'], inplace=True)

    logger.debug(f'Salvando arquivo {file}.csv')
    df.to_csv(f'RESULT/{file}.csv', index=False, sep=';')

def PC():
    logger.debug('Iniciando PROCESSADOR DE CONSULTAS')
    ini = time.time()
    read_config('PC.CFG')
    fim = time.time()
    logger.debug('Fim do PROCESSADOR DE CONSULTAS')
    logger.debug(f'Tempo de execução do PROCESSADOR DE CONSULTAS: {fim-ini:.2f}s')