"""
Módulo: session_store

Responsabilidad:
Gestionar sesiones temporales en memoria para desarrollo.

Cada sesión guarda:
- lead_state
- conversation_history

Importante:
- se pierde al reiniciar el servidor
- sirve para pruebas multi-turn
"""

from typing import Any, Dict


SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    """
    Obtiene una sesión existente o crea una nueva.

    Args:
        session_id (str): identificador único de sesión.

    Returns:
        Dict[str, Any]: sesión con lead_state e historial.
    """
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = {
            "lead_state": None,
            "conversation_history": []
        }

    return SESSION_STORE[session_id]


def get_existing_session(session_id: str) -> Dict[str, Any] | None:
    """
    Obtiene una sesión existente sin crear una nueva.

    Útil para consultar conversaciones desde el dashboard.
    """
    return SESSION_STORE.get(session_id)

def update_session(
    session_id: str,
    lead_state: Dict[str, Any],
    conversation_history: list
) -> None:
    """
    Actualiza el contenido completo de una sesión.

    Args:
        session_id (str): identificador de sesión.
        lead_state (Dict[str, Any]): estado actualizado del lead.
        conversation_history (list): historial actualizado.
    """
    SESSION_STORE[session_id] = {
        "lead_state": lead_state,
        "conversation_history": conversation_history
    }


def clear_session(session_id: str) -> None:
    """
    Elimina una sesión específica.

    Args:
        session_id (str): identificador de sesión.
    """
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]


def clear_all_sessions() -> None:
    """
    Elimina todas las sesiones activas.
    """
    SESSION_STORE.clear()