# agentes/code_generation_agent.py

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
import textwrap
import json
import builtins 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from core.memory import CognitiveMemory

# Lista de módulos permitidos (whitelist) para importação
ALLOWED_IMPORTS = {'pandas', 'numpy', 'matplotlib'}

class SecurityException(Exception):
    """Exceção customizada para violações de segurança."""
    pass

def restricted_import(name: str, globals: Dict[str, Any] = None, locals: Dict[str, Any] = None, fromlist: List[str] = (), level: int = 0):
    """
    Uma versão segura da função __import__ que permite apenas módulos de uma whitelist.
    """
    if name.split('.')[0] not in ALLOWED_IMPORTS:
        raise SecurityException(f"A importação do módulo '{name}' não é permitida.")
    return builtins.__import__(name, globals, locals, fromlist, level)

# Dicionário de funções seguras, incluindo a função de importação restrita.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate, 
    "float": float, "int": int, "isinstance": isinstance, "len": len, "list": list, 
    "max": max, "min": min, "print": print, "range": range, "round": round, 
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "__import__": restricted_import
}

class CodeGenerationAgent:
    """
    Agente responsável por gerar e corrigir código Python para análise de dados.
    """
    def __init__(self, llm: BaseChatModel, data_passport: dict, catalog: dict, memory: 'CognitiveMemory'):
        """
        Inicializa o agente com as dependências necessárias.

        Args:
            llm (BaseChatModel): A instância do modelo de linguagem a ser usado.
            data_passport (dict): O dicionário contendo o perfil dos dados carregados.
            catalog (dict): O dicionário de dataframes, com os nomes das tabelas como chaves.
            memory (CognitiveMemory): A instância da memória cognitiva da sessão.
        """
        self.llm = llm
        self.data_passport = data_passport
        self.catalog = catalog
        self.memory = memory
        self.last_code = ""

    def _get_initial_prompt(self) -> SystemMessage:
        """ Constrói o prompt para a primeira tentativa de geração de código. """
        history = self.memory.get_history_summary()
        
        schema_summary = []
        # Pega o nome da primeira tabela para usar como um exemplo dinâmico e realista no prompt
        example_table_name = next(iter(self.catalog.keys()), 'nome_da_tabela_exemplo')

        for table_name, df in self.catalog.items():
            columns = ", ".join(f"`{col}`" for col in df.columns)
            schema_summary.append(f"- Tabela `{table_name}`: Colunas: [{columns}]")
        schema_str = "\n".join(schema_summary)
        
        prompt = f"""
        Você é um agente de análise de dados de elite expert em Python. Sua tarefa é gerar código Python robusto e bem formatado para uma função 'solve'.

        **REGRAS DE EXECUÇÃO E ROBUSTEZ:**
        1.  **CRÍTICO:** Todas as declarações de `import` DEVEM estar DENTRO da função `solve` para garantir o escopo correto.
        2.  Você PODE e DEVE usar `import` para as bibliotecas `pandas`, `numpy`, e `matplotlib`.
        3.  QUALQUER TENTATIVA de importar um módulo que não seja `{', '.join(ALLOWED_IMPORTS)}` irá falhar.
        4.  SEMPRE crie figuras com um tamanho razoável, por exemplo: `plt.figure(figsize=(10, 6))`.
        5.  SEJA DEFENSIVO: Para operações matemáticas, garanta que a coluna é numérica usando `pd.to_numeric(df['coluna'], errors='coerce')`.
        6.  Para formatar tabelas como texto, use `.to_string()` para evitar dependências externas.
        7.  A saída da sua função DEVE ser uma tupla: `(texto: str, tabela: pd.DataFrame|None, figura: plt.Figure|None)`.

        **ESQUEMA DOS DADOS DISPONÍVEIS:**
        {schema_str}

        **HISTÓRICO DE ANÁLISE (O que já foi feito nesta sessão):**
        {history}
        
        **ESTRUTURA OBRIGATÓRIA DA FUNÇÃO DE RESPOSTA:**
        ```python
        def solve(catalog, question):
            # As importações DEVEM estar aqui dentro.
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Seu código de análise vai aqui.
            # Lembre-se de usar o nome da tabela do ESQUEMA acima.
            df = catalog['{example_table_name}'] # Exemplo de acesso ao DataFrame
            
            text_output = "# Análise Concluída"
            table_output = None
            figure_output = None

            return (text_output, table_output, figure_output)
        ```
        Gere o código completo para a função `solve`.
        """
        return SystemMessage(content=textwrap.dedent(prompt))

    def _get_correction_prompt(self, failed_code: str, error_message: str) -> SystemMessage:
        """ Constrói o prompt para pedir a correção de um código que falhou. """
        prompt = f"""
        Você é um engenheiro de software de elite expert em depuração.
        O código Python que você gerou na tentativa anterior falhou. Sua tarefa é analisar o erro, corrigir o código e fornecer apenas a versão corrigida.

        **CÓDIGO COM ERRO:**
        ```python
        {failed_code}
        ```

        **MENSAGEM DE ERRO:**
        ```
        {error_message}
        ```

        **INSTRUÇÕES:**
        - Analise a mensagem de erro para entender a causa raiz (ex: erro de tipo, método inexistente, coluna não encontrada, escopo de importação).
        - Reescreva a função `solve` completamente, aplicando a correção. Lembre-se que os imports devem estar DENTRO da função.
        - Não escreva explicações, apenas o bloco de código da função `solve` corrigida.
        - Mantenha todas as regras de segurança e formato da tentativa original.

        Agora, forneça o código corrigido.
        """
        return SystemMessage(content=textwrap.dedent(prompt))

    def generate_code(self, query: str) -> str:
        """ Gera a primeira versão do código. """
        prompt = self._get_initial_prompt()
        messages = [prompt, HumanMessage(content=f"Pergunta do Usuário: \"{query}\"")]
        response = self.llm.invoke(messages)
        code = response.content.strip().removeprefix("```python").removesuffix("```").strip()
        self.last_code = code
        return code

    def correct_code(self, failed_code: str, error_message: str) -> str:
        """ Gera uma versão corrigida do código com base no erro. """
        prompt = self._get_correction_prompt(failed_code, error_message)
        messages = [prompt, HumanMessage(content="Por favor, corrija o código.")]
        response = self.llm.invoke(messages)
        code = response.content.strip().removeprefix("```python").removesuffix("```").strip()
        self.last_code = code
        return code