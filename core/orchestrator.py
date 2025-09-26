# core/orchestrator.py

from __future__ import annotations
from typing import TYPE_CHECKING, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from core.memory import CognitiveMemory
from agentes.code_generation_agent import CodeGenerationAgent, SAFE_BUILTINS

if TYPE_CHECKING:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

class Orchestrator:
    """
    O cérebro do sistema, responsável por orquestrar a geração e execução de código,
    incluindo o ciclo de auto-correção.
    """
    def __init__(self, llm: BaseChatModel, catalog: Dict, data_passport: Dict):
        """
        Inicializa o Orquestrador.

        Args:
            llm (BaseChatModel): A instância do modelo de linguagem a ser usado.
            catalog (dict): O dicionário de dataframes (tabelas de dados).
            data_passport (dict): O dicionário contendo o perfil dos dados carregados.
        """
        self.llm = llm
        self.catalog = catalog
        self.data_passport = data_passport
        self.memory = CognitiveMemory(session_id="default")
        self.code_agent = CodeGenerationAgent(llm, data_passport, catalog, self.memory)

    def _synthesize(self) -> Dict:
        """
        Gera uma conclusão com base no histórico da sessão.

        Returns:
            dict: Um dicionário contendo o texto da conclusão e metadados.
        """
        history = self.memory.get_history_summary()
        if "Nenhuma análise" in history:
            return {"text": "Nenhuma análise foi realizada ainda para uma conclusão.", "summary": "Tentativa de síntese sem histórico."}

        prompt = f"Você é um cientista de dados. Baseado no histórico de análises, escreva uma conclusão.\n\nHistórico:\n{history}"
        response = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Gere a conclusão.")])
        return {"text": response.content, "summary": "Síntese final gerada.", "figures": [], "table": None}

    def _execute_code(self, code: str, query: str) -> Dict:
        """
        Tenta executar o código em um ambiente seguro e retorna os artefatos.
        Esta implementação utiliza um escopo unificado para garantir que os imports
        no topo do script sejam visíveis para a função 'solve'.

        Args:
            code (str): O código Python a ser executado.
            query (str): A pergunta original do usuário.

        Returns:
            dict: Um dicionário contendo os artefatos da análise.
        
        Raises:
            Exception: Qualquer exceção que ocorra durante a execução do código.
        """
        scope = {"__builtins__": SAFE_BUILTINS}
        exec(code, scope)
        
        if "solve" not in scope:
            raise RuntimeError("Função 'solve' não foi definida pelo código gerado.")

        solve_func = scope["solve"]
        text, table, fig = solve_func({k: v.copy() for k, v in self.catalog.items()}, query)
        
        return {
            "text": text, 
            "table": table, 
            "figures": [fig] if fig else [], 
            "summary": f"Gerou e executou código para: '{query}'.",
        }

    def handle_query(self, query: str) -> Dict:
        """
        Ponto de entrada principal que gerencia o ciclo completo de análise,
        incluindo a auto-correção em caso de falhas.

        Args:
            query (str): A pergunta em linguagem natural feita pelo usuário.

        Returns:
            dict: Um dicionário contendo os artefatos finais da análise.
        """
        synthesis_keywords = ["conclusão", "resumo", "resuma", "insights gerais", "conclua"]
        if any(keyword in query.lower() for keyword in synthesis_keywords):
            result = self._synthesize()
            agent_name = "Orchestrator(Synthesizer)"
        else:
            max_retries = 3 
            
            code = self.code_agent.generate_code(query)

            for attempt in range(max_retries + 1):
                try:
                    result = self._execute_code(code, query)
                    agent_name = f"CodeGenerationAgent (Tentativa {attempt + 1})"
                    result['code'] = self.code_agent.last_code
                    break
                
                except Exception as e:
                    error_message = f"{type(e).__name__}: {e}"
                    
                    if attempt < max_retries:
                        code = self.code_agent.correct_code(code, error_message)
                    else:
                        result = {
                            "text": f"O agente não conseguiu corrigir o próprio código após {max_retries + 1} tentativas. Erro final: {error_message}",
                            "table": None,
                            "figures": [],
                            "summary": f"Falha final na auto-correção para: '{query}'.",
                            "code": self.code_agent.last_code
                        }
                        agent_name = "CodeGenerationAgent (Falha Final)"
                        break
        
        result['agent_name'] = agent_name
        
        self.memory.add_finding(
            query=query,
            agent_name=result.get("agent_name", "N/A"),
            summary=result['summary'],
            artifacts={"has_table": result.get("table") is not None, "has_figure": bool(result.get("figures"))}
        )
        return result