from ProcessadorDeConsultas import PC
from GeradorListaInvertida import GLI
from Indexador import INDEX
from Buscador import BUSCA
from Avaliacao import AVALIA

import time

import logging, logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('loggerBMT')

if __name__ == "__main__":
    logger.debug('------------- SISTEMA DE RECUPERAÇÃO SEGUNDO O MODELO VETORIAL -------------')
    ini = time.time()
    PC() # PROCESSADOR DE CONSULTAS
    GLI() # GERADOR DE LISTA INVERTIDA
    INDEX() # INDEXADOR
    BUSCA() # BUSCADOR
    AVALIA() #AVALIAÇÃO
    fim = time.time()
    logger.debug('----------------------------------------------------------------------------')
    logger.debug(f'Tempo total de execução: {fim-ini:.2f}s')