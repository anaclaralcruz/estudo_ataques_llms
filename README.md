# LLM Privacy and Security Attacks - Reproduction of LLM-PBE (Open-Source Focus)

Este repositório contém o código desenvolvido como parte do projeto final da disciplina de **Processamento de Linguagem Natural** do Mestrado.

O objetivo do trabalho foi **reproduzir e adaptar parcialmente ataques de segurança e privacidade descritos no artigo [LLM-PBE: Assessing Data Privacy in Large Language Models](https://arxiv.org/abs/2408.12787)**, com foco em **modelos de linguagem open-source**.

## Sobre o projeto

- O estudo original LLM-PBE propõe um toolkit para avaliar riscos de privacidade em LLMs.
- Este projeto **adapta os princípios do LLM-PBE para um cenário open-source**, explorando os ataques:
    - Prompt Leakage Attack (PLA)
    - Data Extraction Attack (DEA)
    - Jailbreak Attack (JA)
- Foram utilizados os modelos:
    - GPT-J 6B
    - Mistral 7B
- As bases utilizadas foram:
    - Prompts BlackFriday (PLA)
    - Emails Enron (DEA)
    - harmful_behaviors.csv (JA)

## Estrutura do repositório

- `pla_attack.py` → implementação do Prompt Leakage Attack
- `dea_attack.py` → implementação do Data Extraction Attack
- `ja_attack.py` → implementação do Jailbreak Attack

Cada script segue o mesmo pipeline geral:

1. Geração de prompts para o ataque
2. Chamada ao modelo com `.generate()`
3. Processamento da resposta
4. Cálculo da FuzzRate
5. (JA) Revisão humana complementar

## Como rodar

### Requisitos

- Python >= 3.8
- Transformers
- HuggingFace Hub
- RapidFuzz
- SentencePiece
- Bitsandbytes (se usar quantização opcional)

### Instalação

```bash
pip install transformers accelerate bitsandbytes sentencepiece rapidfuzz huggingface_hub
