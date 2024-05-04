import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import math

import logging, logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('loggerBMT')

bool_stemmer = False

# Ler um arquivo de configuração
def read_config(config_file):
    global bool_stemmer
    logger.debug(f'Lendo arquivo de configuração ({config_file})')
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('RESULTADOS'):
                file = (line.split('=')[1])[1:-1]

                if((file.split('-')[1])[:-4] == 'STEMMER'):
                    bool_stemmer = True

                df_resultados = pd.read_csv(file,sep=';',dtype={'Query': str})
                df_resultados['Results'] = df_resultados['Results'].apply(lambda x: eval(x))

            elif line.startswith('ESPERADOS'):
                file = (line.split('=')[1])[1:-1]
                df_esperados = pd.read_csv(file,sep=';',dtype={'QueryNumber': str})

    # Gráfico de 11 pontos de precisão e recall
    eleven_points_precision_recall(df_resultados, df_esperados)
    
    # F1
    F1(df_resultados,df_esperados)
    
    # Precision@5
    precision_at_k(df_resultados,df_esperados,5)
    
    # Precision@10
    precision_at_k(df_resultados, df_esperados,10)

    # Histograma de R-Precision (comparativo)
    r_precision(df_resultados,df_esperados)

    # MAP
    MAP(df_resultados, df_esperados)

    # MRR
    MRR(df_resultados, df_esperados)

    # Discounted Cumulative Gain (médio)
    DCG(df_resultados, df_esperados)

    # Normalized Discounted Cumulative Gain
    NDCG(df_resultados, df_esperados)

# Precision
# P = Relevantes Recuperados / Total Recuperados
# Recall
# R = Relevantes Recuperados / Total de itens relevantes
def eleven_points_precision_recall(df_resultados, df_esperados):
    logger.debug('Obtendo Gráfico de 11 pontos de Precisão e Recall')
    precision_recall = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        recuperados = [int(r[1]) for r in results]
        relevantes = df_esperados[df_esperados['QueryNumber'] == query]['DocNumber'].tolist()
        relevantes_recuperados = [x for x in recuperados if x in relevantes]

        P = len(relevantes_recuperados) / len(recuperados)
        R = len(relevantes_recuperados) / len(relevantes)

        precision_recall.append((query,P,R))

    df = pd.DataFrame(precision_recall, columns=['Query', 'Precision', 'Recall'])
    df = df.sort_values(by='Recall',ascending=False)

    recalls = np.linspace(0, 1, 11)
    precisions = []
    
    for rp in recalls:
        pp = df[df['Recall'] >= rp]['Precision'].max()
        
        if len(df[df['Recall'] >= rp]['Precision']) == 0:
            precisions.append(0)
        else:
            precisions.append(pp)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o', linestyle='-')
    plt.title('Gráfico de 11 Pontos de Precisão e Recall')
    plt.xlabel('Recall')
    plt.xlim(-0.05, 1.05)
    plt.ylabel('Precision')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando Gráfico de 11 pontos de Precisão e Recall em arquivo CSV (11pontos-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df_eleven_points = pd.DataFrame({'Recall': recalls, 'Precision': precisions})
    df_eleven_points.to_csv(f"AVALIA/11pontos-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')
   
    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando Gráfico de 11 pontos de Precisão e Recall em arquivo PDF (11pontos-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/11pontos-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

# F1
# F1 = 2PR / (P+R)
def F1(df_resultados, df_esperados):
    logger.debug('Obtendo F1')
    f1 = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        recuperados = [int(r[1]) for r in results]
        relevantes = df_esperados[df_esperados['QueryNumber'] == query]['DocNumber'].tolist()
        relevantes_recuperados = [x for x in recuperados if x in relevantes]

        P = len(relevantes_recuperados) / len(recuperados)
        R = len(relevantes_recuperados) / len(relevantes)

        if(P+R > 0):
            F1 = (2*P*R) / (P+R)
        else:
            F1 = 0

        f1.append((query,F1))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando F1 em arquivo CSV (f1-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(f1, columns=['Query', 'F1'])
    df.to_csv(f"AVALIA/f1-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')
    
    df = df.sort_values(by='F1',ascending=False)

    thresholds = np.linspace(0, 1, 11)
    percentage = []

    for tp in thresholds:
        fp = len(df[df['F1'] >= tp])/len(df['F1'])
        percentage.append(fp)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, percentage, marker='o', linestyle='-')
    plt.title('Gráfico de 11 Pontos de F1 (%)')
    plt.xlabel('F1')
    plt.xlim(-0.05, 1.05)
    plt.ylabel('%')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando F1 em arquivo PDF (f1-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/f1-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

# Precision@K
# Porcentagem de documentos relevantes no top-K escolhidos pelo modelo
def precision_at_k(df_resultados, df_esperados,k):
    logger.debug(f'Obtendo Precision@{k}')
    precision_k = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        recuperados_k = [int(r[1]) for r in results][:k]

        relevantes = df_esperados[df_esperados['QueryNumber'] == query]
        relevantes = relevantes.sort_values(by='DocVotes',ascending=False)
        relevantes = relevantes['DocNumber'].tolist()

        relevantes_recuperados_k = [x for x in recuperados_k if x in relevantes]

        P_k = len(relevantes_recuperados_k) / k
        precision_k.append((query,P_k))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando Precision@{k} em arquivo CSV (precision@{k}-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(precision_k, columns=['Query', 'Precision'])
    df.to_csv(f"AVALIA/precision@{k}-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')
    
    df = df.sort_values(by='Precision',ascending=False)

    thresholds = np.linspace(0, 1, 11)
    percentage = []
    
    for tp in thresholds:
        pp = len(df[df['Precision'] >= tp])/len(df['Precision'])
        percentage.append(pp)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, percentage, marker='o', linestyle='-')
    plt.title(f'Gráfico de 11 Pontos de Precision@{k} (%)')
    plt.xlabel(f'Precision@{k}')
    plt.xlim(-0.05, 1.05)
    plt.ylabel('%')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando F1 em arquivo PDF (precision@{k}-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/precision@{k}-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

# R-Precision
# Se sabemos um conjunto de documentos relevantes de tamanho Rel, então calcular o número de documentos relevantes encontrados nos Rel primeiros documentos
# R-Precision = Recuperados Relevantes / Total de Relevantes
def r_precision(df_resultados, df_esperados):
    logger.debug('Obtendo Histograma de R-Precision')
    rprecision = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        relevantes = df_esperados[df_esperados['QueryNumber'] == query]['DocNumber'].tolist()
        recuperados_rel = [int(r[1]) for r in results][:len(relevantes)]
        relevantes_recuperados = [x for x in recuperados_rel if x in relevantes]

        RP = len(relevantes_recuperados) / len(relevantes)
        rprecision.append((query,RP))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando R-Precision em arquivo CSV (rprecision-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(rprecision, columns=['Query', 'R-Precision'])
    df.to_csv(f"AVALIA/rprecision-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')
    
    # Para gerar o Histograma R-Precision (comparativo), é necessário dois algoritmos diferentes.
    # Aqui, usaremos STEMMER e NOSTEMMER 
    # Só será possível obter o histograma caso RESULTADOS-STEMMER.csv e RESULTADOS-NOSTEMMER.csv existirem.
    try:
        # A=STEMMER, B=NOSTEMMER
        if(bool_stemmer):
            df_resultados_comp = pd.read_csv('RESULT/RESULTADOS-NOSTEMMER.csv',sep=';',dtype={'Query': str})
        # A=NOSTEMMER, B=STEMMER
        else:
            df_resultados_comp = pd.read_csv('RESULT/RESULTADOS-STEMMER.csv',sep=';',dtype={'Query': str})
        
        df_resultados_comp['Results'] = df_resultados_comp['Results'].apply(lambda x: eval(x))

        rprecision_comp = []

        for query, results in zip(df_resultados_comp['Query'],df_resultados_comp['Results']):
            relevantes = df_esperados[df_esperados['QueryNumber'] == query]['DocNumber'].tolist()
            recuperados_rel = [int(r[1]) for r in results][:len(relevantes)]
            relevantes_recuperados = [x for x in recuperados_rel if x in relevantes]

            RP = len(relevantes_recuperados) / len(relevantes)
            rprecision_comp.append((query,RP))

        RP_AB = [a[1]-b[1] for a,b in zip(rprecision,rprecision_comp)]
        query_number = [int(x[0]) for x in rprecision]

        plt.figure(figsize=(8, 6))
        plt.bar(query_number, RP_AB)
        plt.title(f"Histograma de R-Precision {'STEMMER/NOSTEMMER' if bool_stemmer else 'NOSTEMMER/STEMMER'}")
        plt.xlabel('Query Number')
        plt.ylabel("R-Precision")
        plt.grid(True)
        
        # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
        logger.debug(f"Salvando R-Precision em arquivo PDF (rprecision-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
        plt.savefig(f"AVALIA/rprecision-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")
        
    except:
        logger.error('Não foi possível obter Histograma de R-Precision. RESULTADOS-STEMMER.csv ou RESULTADOS-NOSTEMMER.csv não existe.')

# Mean Average Precision (MAP)
# Média do valor da precisão para os k documentos de topo, cada vez que um documento relevante é recuperado
def MAP(df_resultados, df_esperados):
    logger.debug('Obtendo MAP (Mean Average Precision)')
    average_precision = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        relevantes = df_esperados[df_esperados['QueryNumber'] == query]['DocNumber'].tolist()
        recuperados = [int(r[1]) for r in results]

        average_precision_query = []
        num_recuperados_relevantes = 0
        pos_recuperados = 1
        
        for rec in recuperados:
            if(rec in relevantes):
                num_recuperados_relevantes += 1
                average_precision_query.append(num_recuperados_relevantes / pos_recuperados)
            pos_recuperados += 1

        if(len(average_precision_query) != 0):
            average_precision.append((query,sum(average_precision_query)/len(average_precision_query)))
        else:
            average_precision.append((query,0))

    _map = sum(v for k,v in average_precision)/len(average_precision)
    average_precision.append(('MAP',_map))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando MAP em arquivo CSV (map-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(average_precision, columns=['Query', 'Average Precision'])
    df.to_csv(f"AVALIA/map-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')

    df = df.sort_values(by='Average Precision',ascending=False)

    thresholds = np.linspace(0, 1, 11)
    percentage = []

    df = df[:-1] # Tirar linha do MAP
    
    for tp in thresholds:
        app = len(df[df['Average Precision'] >= tp])/len(df['Average Precision'])
        percentage.append(app)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, percentage, marker='o', linestyle='-')
    plt.axvline(x=_map, color='r', linestyle='--', label='MAP')
    plt.title(f'Gráfico de 11 Pontos de Average Precision (%)')
    plt.xlabel('Average Precision')
    plt.xlim(-0.05, 1.05)
    plt.ylabel('%')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()

    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando MAP em arquivo PDF (map-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/map-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

# Mean Reciprocal Rank (MRR)
# Considere a posição K do primeiro documento relevante
# Reciprocal Rank score = 1 / K
# MRR é a média dos RR em várias consultas
def MRR(df_resultados, df_esperados):
    logger.debug('Obtendo MRR (Mean Reciprocal Rank)')
    reciprocal_rank = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        relevantes = df_esperados[df_esperados['QueryNumber'] == query]['DocNumber'].tolist()
        recuperados = [int(r[1]) for r in results]

        pos_recuperados = 1
        for rec in recuperados:
            if(rec in relevantes):
                reciprocal_rank.append((query,1/pos_recuperados))
                break
            pos_recuperados += 1

    mrr = sum(v for k,v in reciprocal_rank)/len(reciprocal_rank)
    reciprocal_rank.append(('MRR',mrr))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando MRR em arquivo CSV (mrr-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(reciprocal_rank, columns=['Query', 'Reciprocal Rank'])
    df.to_csv(f"AVALIA/mrr-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')

    df = df.sort_values(by='Reciprocal Rank',ascending=False)

    thresholds = np.linspace(0, 1, 11)
    percentage = []

    df = df[:-1] # Tirar linha do MRR
    
    for tp in thresholds:
        rrp = len(df[df['Reciprocal Rank'] >= tp])/len(df['Reciprocal Rank'])
        percentage.append(rrp)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, percentage, marker='o', linestyle='-')
    plt.axvline(x=mrr, color='r', linestyle='--', label='MRR')
    plt.title(f'Gráfico de 11 Pontos de Reciprocal Rank (%)')
    plt.xlabel('Reciprocal Rank')
    plt.xlim(-0.05, 1.05)
    plt.ylabel('%')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()

    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando MRR em arquivo PDF (mrr-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/mrr-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

# Discounted Cumulative Gain
# Documentos altamente relevantes são muito mais úteis que documentos marginalmente relevantes
# Quanto maior é a posição de um documento na lista retornada, menos útil ele é, pois há menos chance de ser examinado
def DCG(df_resultados, df_esperados):
    logger.debug('Obtendo DCG (Discounted Cumulative Gain)')
    dcg = []

    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        relevantes = df_esperados[df_esperados['QueryNumber'] == query].set_index('DocNumber')['DocVotes'].to_dict()
        recuperados = [int(r[1]) for r in results]

        dcg_query = 0
        pos_recuperados = 1
        for rec in recuperados:
            if(rec in relevantes):
                vote = relevantes[rec]
                dcg_query += vote/math.log2(pos_recuperados+1)
            pos_recuperados += 1
        
        dcg.append((query,dcg_query))

    mean_dcg = sum(v for k,v in dcg)/len(dcg)
    dcg.append(('Mean DCG',mean_dcg))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando DCG em arquivo CSV (dcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(dcg, columns=['Query', 'DCG'])
    df.to_csv(f"AVALIA/dcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')

    df = df[:-1] # Tirar linha do Mean DCG

    query_number = df['Query'].apply(int).to_list()
    dcg = df['DCG'].to_list()

    plt.figure(figsize=(8, 6))
    plt.bar(query_number, dcg)
    plt.title(f"Histograma de DCG")
    plt.xlabel('Query Number')
    plt.ylabel("DCG")
    plt.axhline(y=mean_dcg, color='r', linestyle='--', label='Mean DCG')
    plt.grid(True)
    plt.legend()

    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando DCG em arquivo PDF (dcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/dcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

# Normalized Discounted Cumulative Gain
# Normaliza o DCG@r pelo DCG@r do ranking ideal (Ideal DCG)
def NDCG(df_resultados, df_esperados):
    logger.debug('Obtendo NDCG (Normalized Discounted Cumulative Gain)')
    
    # DCG (Discounted Cumulative Gain)
    dcg = []
    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        relevantes = df_esperados[df_esperados['QueryNumber'] == query].set_index('DocNumber')['DocVotes'].to_dict()
        recuperados = [int(r[1]) for r in results]

        dcg_query = 0
        pos_recuperados = 1
        for rec in recuperados:
            if(rec in relevantes):
                vote = relevantes[rec]
                dcg_query += vote/math.log2(pos_recuperados+1)
            pos_recuperados += 1
        
        dcg.append((query,dcg_query))

    # IDCG (Ideal Discounted Cumulative Gain)
    idcg = []
    for query, results in zip(df_resultados['Query'],df_resultados['Results']):
        relevantes = df_esperados[df_esperados['QueryNumber'] == query]
        relevantes = relevantes.sort_values(by='DocVotes',ascending=False)
        relevantes = relevantes.set_index('DocNumber')['DocVotes'].to_dict()
        recuperados = [int(r[1]) for r in results]

        idcg_query = 0
        pos_relevantes = 1
        for rel in relevantes:
            vote = relevantes[rel]
            idcg_query += vote/math.log2(pos_relevantes+1)
            pos_relevantes += 1
        
        idcg.append((query,idcg_query))

    # NDCG (Normalized Discounted Cumulative Gain)
    ndcg = []
    for _dcg_,_idcg_ in zip(dcg,idcg):
        query = _dcg_[0]
        dcg_value = _dcg_[1]
        idcg_value = _idcg_[1]
        ndcg.append((query,dcg_value/idcg_value))

    mean_ndcg = sum(v for k,v in ndcg)/len(ndcg)
    ndcg.append(('Mean NDCG',mean_ndcg))

    # Salvar diagrama na forma de um arquivo CSV
    logger.debug(f"Salvando NDCG em arquivo CSV (ndcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv)")
    df = pd.DataFrame(ndcg, columns=['Query', 'NDCG'])
    df.to_csv(f"AVALIA/ndcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.csv", index=False, sep=';')

    df = df[:-1] # Tirar linha do Mean DCG

    query_number = df['Query'].apply(int).to_list()
    ndcg = df['NDCG'].to_list()

    plt.figure(figsize=(8, 6))
    plt.bar(query_number, ndcg)
    plt.title(f"Histograma de NDCG")
    plt.xlabel('Query Number')
    plt.ylabel("NDCG")
    plt.axhline(y=mean_ndcg, color='r', linestyle='--', label='Mean NDCG')
    plt.grid(True)
    plt.legend()

    # Salvar diagrama na forma de um arquivo gráfico (PDF ou outro)
    logger.debug(f"Salvando NDCG em arquivo PDF (ndcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf)")
    plt.savefig(f"AVALIA/ndcg-{'stemmer-2' if bool_stemmer else 'nostemmer-1'}.pdf")

def AVALIA():
    logger.debug('Iniciando AVALIAÇÃO')
    ini = time.time()
    read_config('AVALIA.CFG')
    fim = time.time()
    logger.debug('Fim do AVALIAÇÃO')
    logger.debug(f'Tempo de execução do AVALIAÇÃO: {fim-ini:.2f}s')

# Referências:
# - Slides das Aulas
# - Evaluation Metrics For Information Retrieval (https://amitness.com/posts/information-retrieval-evaluation)