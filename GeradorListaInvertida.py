# GERADOR LISTA INVERTIDA
# A função desse módulo é criar as listas invertidas simples.

import pandas as pd
import time
import re
from collections import defaultdict

import logging, logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('loggerBMT')

import xmltodict
from lxml import etree

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Ler um arquivo de configuração
def read_config(config_file):
    logger.debug(f'Lendo arquivo de configuração ({config_file})')
    list_xml = []
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('LEIA'):
                file = (line.split('=')[1])[1:-1]
                list_xml.append(read_xml(file))

            elif line.startswith('ESCREVA'):
                file = (line.split('=')[1])[1:-1]
                inverted_index(list_xml,file)

# Ler um arquivo em formato XML
def read_xml(xml_file):
    logger.debug(f'Lendo arquivo XML ({xml_file})')
    parser = etree.XMLParser(dtd_validation=True)
    tree = etree.parse(f'data/{xml_file}', parser)
    dict = xmltodict.parse(etree.tostring(tree))
    len_dict = len(dict['FILE']['RECORD'])
    logger.debug(f'Total de documentos em {xml_file}: {len_dict}')
    return dict['FILE']['RECORD']

def inverted_index(list_xml,file):
    logger.debug('Gerando lista invertida')

    invertedIndex = defaultdict(list)

    for xml in list_xml:
        df = pd.DataFrame(xml)
        
        df['ABSTRACT'].fillna(df['EXTRACT'], inplace=True)
        df = df[['RECORDNUM', 'ABSTRACT']]
        df = df.dropna()

        df['RECORDNUM'] = df['RECORDNUM'].apply(lambda x: int(x))

        # remover pontuação, remover números, tokenização, lematização, remover stopwords, remover tokens de tamanho menor que 5, maiúscula
        df['ABSTRACT'] = df['ABSTRACT'].apply(lambda x: remove_punct(str(x)))
        df['ABSTRACT'] = df['ABSTRACT'].apply(remove_numbers)
        df['ABSTRACT'] = df['ABSTRACT'].apply(word_tokenize)
        df['ABSTRACT'] = df['ABSTRACT'].apply(lemmatizer)
        df['ABSTRACT'] = df['ABSTRACT'].apply(remove_stop_words)
        df['ABSTRACT'] = df['ABSTRACT'].apply(remove_len)
        df['ABSTRACT'] = df['ABSTRACT'].apply(upper)

        for recordnum, abstract in zip(df['RECORDNUM'], df['ABSTRACT']):
            for t in abstract:
                invertedIndex[t].append(recordnum)

    invertedIndex = dict(sorted(invertedIndex.items()))

    len_invertedIndex = len(invertedIndex)
    logger.debug(f'Total de tokens na lista invertida: {len_invertedIndex}')

    logger.debug(f'Salvando arquivo {file}.csv')
    df_invertedIndex = pd.DataFrame(list(invertedIndex.items()), columns=['TOKEN', 'LIST_DOCUMENTS'])
    df_invertedIndex.to_csv(f'RESULT/{file}.csv', index=False, header=False, sep=';')

# Remover números
def remove_numbers(string):
    return re.sub(r'\d+', ' ', string)

# Remover stopwords
def remove_stop_words(tokens): 
    stop_words = set(stopwords.words('english'))
    stop_words.add('patient')
    ft = [t for t in tokens if t.lower() not in stop_words] 
    return ft

# Remover pontuação
def remove_punct(string):
    return re.sub(r'[^\w\s]', ' ', string)

# Remover tamanho
def remove_len(tokens):
    ft = [t for t in tokens if len(t)>=5] 
    return ft

# Lemmatization
def lemmatizer(tokens):
    lt = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return lt

# Upper
def upper(tokens):
    u = [str.upper(t) for t in tokens]
    return u

def GLI():
    logger.debug('Iniciando GERADOR LISTA INVERTIDA')
    ini = time.time()
    read_config('GLI.CFG')
    fim = time.time()
    logger.debug('Fim do GERADOR LISTA INVERTIDA')
    logger.debug(f'Tempo de execução do GERADOR LISTA INVERTIDA: {fim-ini:.2f}s')