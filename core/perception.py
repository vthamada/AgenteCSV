# core/perception.py

import io
import re
import zipfile
from typing import List, Tuple, Dict, Any
import pandas as pd

def _read_csv_any(file_like: io.BytesIO) -> pd.DataFrame:
    """
    Lê um objeto de bytes de um arquivo CSV, tentando diferentes encodings.

    Args:
        file_like (io.BytesIO): O conteúdo do arquivo CSV em memória.

    Returns:
        pd.DataFrame: Um DataFrame do pandas criado a partir do arquivo.
    """
    try:
        file_like.seek(0)
        return pd.read_csv(file_like, sep=None, engine="python", encoding="utf-8")
    except Exception:
        file_like.seek(0)
        return pd.read_csv(file_like, sep=None, engine="python", encoding="latin-1")

def load_catalog_from_uploads(files: List[Tuple[str, bytes]]) -> Dict[str, pd.DataFrame]:
    """
    Carrega múltiplos arquivos, extrai CSVs (inclusive de ZIPs) e os organiza em um catálogo.

    Args:
        files (List[Tuple[str, bytes]]): Uma lista de tuplas, onde cada uma contém o nome e o conteúdo de um arquivo.

    Returns:
        Dict[str, pd.DataFrame]: Um dicionário onde as chaves são nomes de tabela sanitizados e os valores são os DataFrames.
    """
    catalog: Dict[str, pd.DataFrame] = {}
    
    def _safe_name(stem: str) -> str:
        """Cria um nome de tabela seguro e único a partir de um nome de arquivo."""
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", stem).strip("_").lower()
        if not s: s = "tabela"
        base, i = s, 2
        while s in catalog:
            s = f"{base}_{i}"; i += 1
        return s

    for fname, data in files:
        lower = fname.lower()
        if lower.endswith(".csv"):
            df = _read_csv_any(io.BytesIO(data))
            catalog[_safe_name(re.sub(r"\.csv$", "", fname, flags=re.I))] = df
        elif lower.endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(data))
            for m in zf.namelist():
                if m.lower().endswith(".csv") and not m.startswith("__MACOSX"):
                    df = _read_csv_any(io.BytesIO(zf.read(m)))
                    catalog[_safe_name(re.sub(r"\.csv$", "", m.split("/")[-1], flags=re.I))] = df
    
    if not catalog: 
        raise RuntimeError("Nenhum arquivo CSV válido foi encontrado nos uploads.")
    
    return catalog

def create_data_passport(catalog: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Gera um dicionário de metadados ("Passaporte") que descreve a estrutura de cada DataFrame no catálogo.

    Args:
        catalog (Dict[str, pd.DataFrame]): O catálogo de DataFrames carregados.

    Returns:
        Dict[str, Any]: Um dicionário contendo o perfil detalhado de cada tabela.
    """
    passport = {"tables": {}}
    for name, df in catalog.items():
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        passport["tables"][name] = {
            "shape": f"{df.shape[0]} linhas x {df.shape[1]} colunas",
            "columns": df.columns.tolist(),
            "info": info_str,
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude='number').columns.tolist(),
        }
    return passport