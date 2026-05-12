"""
Lead Storage Module

Este módulo se encarga de persistir los leads detectados por el agente AI
durante cada interacción.

Actualmente usa almacenamiento en archivo JSON local como MVP.

Diseñado para ser reemplazable por:
- Base de datos (PostgreSQL, MongoDB)
- Google Sheets
- CRM externo

Autor: Giovanny
Proyecto: Premium Car AI Agent
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


# Ruta del archivo donde se almacenan los leads
FILE_PATH = Path("data/leads.json")


def load_leads() -> List[Dict]:
    """
    Carga todos los leads almacenados desde el archivo JSON.

    Returns:
        List[Dict]: lista de leads existentes.
    """
    if FILE_PATH.exists():
        try:
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_lead(session_id: str, lead_data: Dict) -> None:
    """
    Guarda el estado del lead asociado a una sesión en el archivo JSON.

    Responsabilidad:
    - Persistir cada interacción del agente (no solo leads finales).
    - Asociar cada registro a una conversación mediante `session_id`.

    Flujo:
    1. Asegura que el directorio exista.
    2. Carga leads existentes.
    3. Enriquece el lead con metadata:
       - session_id: identifica la conversación
       - timestamp: momento de la interacción (UTC)
       - source: origen del lead
    4. Agrega el nuevo registro.
    5. Persiste en disco.

    Args:
        session_id (str):
            Identificador único de la conversación.
            Permite agrupar múltiples interacciones del mismo cliente.

        lead_data (Dict):
            Estado estructurado del lead generado por el agente.

    Example:
        save_lead(
            session_id="session-123",
            lead_data={
                "city": "Bogotá",
                "budget_range": "320 millones COP",
                "current_vehicle": "BMW X1"
            }
        )

    Notes:
        - Este diseño guarda TODAS las interacciones (no hace merge).
        - Permite análisis posterior del comportamiento del cliente.
        - Base para futura consolidación por session_id.
    """

    # -----------------------------
    # 1. Asegurar directorio
    # -----------------------------
    FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 2. Cargar leads existentes
    # -----------------------------
    leads = load_leads()

    # -----------------------------
    # 3. Enriquecer lead
    # -----------------------------
    lead_data_enriched = {
        **lead_data,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "ai_agent"
    }

    # -----------------------------
    # 4. Consolidar por session_id
    # -----------------------------
    # Si ya existe un lead para esta sesión, lo actualizamos.
    # Si no existe, lo agregamos como nuevo.
    existing_index = next(
        (
            index
            for index, lead in enumerate(leads)
            if lead.get("session_id") == session_id
        ),
        None
    )

    if existing_index is not None:
        leads[existing_index] = {
            **leads[existing_index],
            **lead_data_enriched
        }
    else:
        leads.append(lead_data_enriched)
    
    # -----------------------------
    # 5. Guardar archivo actualizado
    # -----------------------------
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(leads, f, indent=2, ensure_ascii=False)