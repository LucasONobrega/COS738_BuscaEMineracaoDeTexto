# COS738 - Busca e Mineração de Texto

Este repositório contém todos os arquivos utilizados no trabalho individual **Implementação de um Sistema de Recuperação em Memória segundo o Modelo Vetorial** da disciplina Busca e Mineração de Texto (COS738) do Programa de Engenharia de Sistemas e Computação (COPPE/PESC) da Universidade Federal do Rio de Janeiro (UFRJ).

## Visão Geral

O trabalho consiste na criação de um sistema de recuperação da informação utilizando a base CysticFibrosis2. O modelo do sistema a ser implementado é composto pelos seguintes componentes:

- **Processador de Consultas**: O objetivo desse módulo é transformar o arquivo de consultas fornecido ao padrão de palavras que estamos utilizando.
- **Gerador de Lista Invertida**: A função desse módulo é criar as listas invertidas simples.
- **Indexador**: A função desse módulo é criar o modelo vetorial, dadas as listas invertidas simples.
- **Buscador**: O objetivo desse módulo é obter os resultados de um conjunto de buscas em um modelo salvo.
- **Avaliação**: O objetivo desse múdulo é avaliar o sistema de recuperação de informação.

## Como Usar

Para executar o sistema, basta rodar o arquivo `main.py`.
