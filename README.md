# Agente de Análise de CSV

## 1. Visão Geral

Este projeto implementa um agente de Inteligência Artificial avançado, capaz de realizar Análise Exploratória de Dados (E.D.A.) em qualquer arquivo CSV fornecido pelo usuário. A interação é feita através de linguagem natural, onde o agente interpreta a pergunta, gera e executa código Python de forma segura para produzir respostas, tabelas e visualizações.

O sistema foi desenvolvido como solução para o **Desafio Extra** do *Institut d'Intelligence Artificielle Appliquée*, com foco em robustez, resiliência, segurança e excelência na experiência do usuário.

## 2. Arquitetura do Sistema

A arquitetura final do sistema é a **Arquitetura Híbrida Direta (AHD) com Ciclo de Auto-Correção**. Esta abordagem combina a flexibilidade da geração de código por IA com a segurança de um ambiente controlado e a inteligência de um sistema que aprende com seus próprios erros.

### Pilares da Arquitetura:

* **Agente Único de Geração de Código:** Um poderoso `CodeGenerationAgent` é o único responsável por todas as tarefas analíticas, simplificando o fluxo de trabalho.
* **Sandbox Seguro:** Todo o código gerado é executado em um ambiente restrito que utiliza um `__import__` customizado para permitir apenas bibliotecas seguras (`pandas`, `numpy`, `matplotlib`), prevenindo a execução de código malicioso.
* **Ciclo de Auto-Correção:** O `Orquestrador` atua como um supervisor. Se o código gerado pelo agente falhar, o orquestrador captura o erro e instrui o agente a analisar a falha e a gerar uma versão corrigida. Este ciclo pode se repetir, tornando o sistema extremamente resiliente.
* **Consciência Contextual:** O sistema gera um "Passaporte de Dados" no momento do upload, informando ao agente sobre o esquema dos dados. Ele também mantém um histórico da conversa para análises contextuais e para a geração de resumos.

## 3. Funcionalidades

* **Upload de Múltiplos CSVs:** Suporte para carregar um ou mais arquivos CSV, ou um único arquivo `.zip` contendo múltiplos CSVs.
* **Análise em Linguagem Natural:** Faça perguntas complexas sobre os dados em português.
* **Geração de Respostas Multimodais:** O agente responde com texto explicativo, tabelas de dados formatadas e gráficos gerados dinamicamente.
* **Interface Interativa:** Interface web construída com Streamlit, permitindo fácil configuração do modelo de LLM, upload de arquivos e interação via chat.
* **Alta Resiliência:** O ciclo de auto-correção permite que o sistema se recupere de erros de programação gerados pela IA.
* **Segurança:** Ambiente de execução `exec()` rigorosamente controlado para prevenir vulnerabilidades.
* **Transparência:** Opção de visualizar o código exato que foi executado para gerar cada resposta.

## 4. Estrutura do Projeto

```
/AgenteCSV/
|-- 📄 README.md                # Este arquivo
|-- 📄 requirements.txt         # Dependências do projeto
|-- 🐍 app.py                    # Aplicação principal Streamlit (Interface)
|-- 🐍 modelos_llm.py            # Fábrica para criar instâncias de LLMs
|
|-- 📂 core/
|   |-- __init__.py
|   |-- orchestrator.py      # Orquestrador com ciclo de auto-correção
|   |-- memory.py            # Gerenciador de memória da sessão
|   |-- perception.py        # Módulo de carregamento de dados e Passaporte
|
|-- 📂 agentes/
|   |-- __init__.py
|   |-- code_generation_agent.py # Agente de geração e correção de código
```

## 5. Guia de Instalação e Execução

### Pré-requisitos
* Python 3.8 ou superior

### Passos para Instalação

1.  **Clone o repositório:**
    ```sh
    git clone [URL_DO_SEU_REPOSITORIO]
    cd seu_projeto
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure as Chaves de API:**
    O sistema buscará as chaves de API das suas variáveis de ambiente. Defina a chave para o provedor que deseja usar (ex: `GEMINI_API_KEY`).
    ```sh
    export GEMINI_API_KEY="SUA_CHAVE_API_AQUI"
    ```
    Alternativamente, você pode inseri-la diretamente na interface da aplicação.

### Execução

Para iniciar a aplicação, execute o seguinte comando na raiz do projeto:

```sh
streamlit run app.py
```
A aplicação será aberta em seu navegador padrão.

## 6. Tecnologias Utilizadas

* **Python:** Linguagem principal do projeto.
* **Streamlit:** Framework para a construção da interface web interativa.
* **LangChain:** Framework para a orquestração e interação com os Modelos de Linguagem (LLMs).
* **Pandas:** Biblioteca para manipulação e análise dos dados.
* **Matplotlib:** Biblioteca para a geração de gráficos e visualizações.