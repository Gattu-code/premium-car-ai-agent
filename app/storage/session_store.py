"""
Módulo: session_store

Responsabilidad:
Gestionar sesiones de conversación para desarrollo/MVP.

Cada sesión guarda:
- lead_state
- conversation_history

Persistencia:
- En memoria para acceso rápido.
- En JSON local para que no se pierda al reiniciar el servidor.

Archivo:
- data/conversations.json
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


SESSION_STORE: Dict[str, Dict[str, Any]] = {}

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CONVERSATIONS_FILE = DATA_DIR / "conversations.json"


def _ensure_data_dir() -> None:
    """
    Asegura que exista la carpeta data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_persisted_sessions() -> Dict[str, Dict[str, Any]]:
    """
    Carga sesiones persistidas desde JSON.

    Si el archivo no existe o está corrupto, devuelve un dict vacío.
    """
    _ensure_data_dir()

    if not CONVERSATIONS_FILE.exists():
        return {}

    try:
        with CONVERSATIONS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except json.JSONDecodeError:
        return {}
    except OSError:
        return {}


def _save_persisted_sessions(data: Dict[str, Dict[str, Any]]) -> None:
    """
    Guarda sesiones en JSON.
    """
    _ensure_data_dir()

    with CONVERSATIONS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def get_session(session_id: str) -> Dict[str, Any]:
    """
    Obtiene una sesión existente o crea una nueva.

    Prioridad:
    1. Memoria.
    2. JSON persistido.
    3. Nueva sesión vacía.

    Args:
        session_id (str): identificador único de sesión.

    Returns:
        Dict[str, Any]: sesión con lead_state e historial.
    """
    if session_id in SESSION_STORE:
        return SESSION_STORE[session_id]

    persisted_sessions = _load_persisted_sessions()

    if session_id in persisted_sessions:
        SESSION_STORE[session_id] = persisted_sessions[session_id]
        return SESSION_STORE[session_id]

    SESSION_STORE[session_id] = {
        "lead_state": None,
        "conversation_history": [],
    }

    return SESSION_STORE[session_id]


def get_existing_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene una sesión existente sin crear una nueva.

    Útil para consultar conversaciones desde el dashboard.
    """
    if session_id in SESSION_STORE:
        return SESSION_STORE[session_id]

    persisted_sessions = _load_persisted_sessions()

    session = persisted_sessions.get(session_id)

    if session:
        normalized_session = {
            "lead_state": None,
            "conversation_history": session.get("conversation_history", []),
        }

        SESSION_STORE[session_id] = normalized_session
        return normalized_session

    return None

def _deduplicate_consecutive_messages(conversation_history: list) -> list:
    """
    Elimina mensajes consecutivos idénticos del mismo rol.

    Evita duplicados accidentales como:
    user: "nos vemos en la cita"
    user: "nos vemos en la cita"
    """
    cleaned = []

    for message in conversation_history:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        content = str(message.get("content", "")).strip()

        if not content:
            continue

        if cleaned:
            last = cleaned[-1]
            last_role = last.get("role", "")
            last_content = str(last.get("content", "")).strip()

            if role == last_role and content == last_content:
                continue

        cleaned.append({
            "role": role,
            "content": content,
        })

    return cleaned

def update_session(
    session_id: str,
    lead_state: Dict[str, Any],
    conversation_history: list,
) -> None:
    """
    Actualiza el contenido completo de una sesión.

    En memoria guarda:
    - lead_state
    - conversation_history

    En data/conversations.json persiste solo:
    - conversation_history

    Motivo:
    El lead_state completo ya se guarda en el storage de leads.
    Persistirlo también en conversations.json genera redundancia
    y posibles inconsistencias.
    """
    
    conversation_history = _deduplicate_consecutive_messages(conversation_history)
    
    session_data = {
        "lead_state": lead_state,
        "conversation_history": conversation_history,
    }

    # Memoria: útil durante la conversación activa
    SESSION_STORE[session_id] = session_data

    # Persistencia: solo historial conversacional
    persisted_sessions = _load_persisted_sessions()
    persisted_sessions[session_id] = {
        "conversation_history": conversation_history,
    }

    _save_persisted_sessions(persisted_sessions)

def clear_session(session_id: str) -> None:
    """
    Elimina una sesión específica de memoria y del JSON.

    Args:
        session_id (str): identificador de sesión.
    """
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]

    persisted_sessions = _load_persisted_sessions()

    if session_id in persisted_sessions:
        del persisted_sessions[session_id]
        _save_persisted_sessions(persisted_sessions)


def clear_all_sessions() -> None:
    """
    Elimina todas las sesiones activas y persistidas.
    """
    SESSION_STORE.clear()
    _save_persisted_sessions({})