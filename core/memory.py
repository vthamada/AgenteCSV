# core/memory.py

from typing import List, Dict, Any, Optional
import datetime

class CognitiveMemory:
    """Gerencia o estado e o histórico da sessão de análise."""

    def __init__(self, session_id: str):
        """
        Inicializa a memória da sessão.

        Args:
            session_id (str): Um identificador único para a sessão de chat.
        """
        self.session_id = session_id
        self.findings: List[Dict[str, Any]] = []

    def add_finding(self, query: str, agent_name: str, summary: str, artifacts: Optional[Dict] = None):
        """
        Registra uma nova descoberta ou análise no histórico da sessão.

        Args:
            query (str): A pergunta original do usuário.
            agent_name (str): O nome do agente ou módulo que processou a pergunta.
            summary (str): Um resumo conciso do que foi feito.
            artifacts (Optional[Dict], optional): Metadados sobre os artefatos gerados (ex: se há tabela ou figura).
        """
        finding = {
            "id": f"f-{len(self.findings) + 1:03d}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "query": query,
            "agent_used": agent_name,
            "summary": summary,
            "artifacts": artifacts or {},
        }
        self.findings.append(finding)

    def get_history_summary(self) -> str:
        """
        Retorna um resumo de todo o histórico de análises da sessão.

        Returns:
            str: Uma string formatada em markdown com o log de todas as análises.
        """
        if not self.findings:
            return "Nenhuma análise foi realizada ainda."
        
        lines = [f"- {f['summary']}" for f in self.findings]
        return "\n".join(lines)