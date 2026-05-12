"""
Módulo: main

Responsabilidad:
Exponer la API principal del proyecto usando FastAPI.

Este módulo publica endpoints para:
- validar que el servicio esté vivo,
- obtener un estado vacío del lead,
- procesar mensajes del agente comercial premium.

Este será el punto de entrada para pruebas manuales,
Swagger y futuras integraciones con n8n.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import json
from app.agents.premium_agent import process_lead_message
from app.storage.session_store import (
    get_session,
    update_session,
    get_existing_session,
    clear_session,
    clear_all_sessions,
)
from app.models.lead_model import LeadModel


# =========================
# CONFIGURACIÓN DE APP
# =========================

app = FastAPI(
    title="Premium Car AI Agent",
    description="API para un agente comercial de vehículos premium con FastAPI + Ollama",
    version="0.1.0"
)


# =========================
# MODELOS DE REQUEST
# =========================

class LeadAgentRequest(BaseModel):
    """
    Modelo de entrada para el endpoint principal del agente.

    Attributes:
        session_id (str):
            Identificador único de la conversación o sesión.

        user_message (str):
            Último mensaje enviado por el usuario.

        lead_state (dict | None):
            Estado acumulado del lead hasta el momento.
            Si no se envía, el sistema crea uno vacío.

        catalog_context (str):
            Contexto opcional del catálogo o portafolio comercial.

        conversation_history (list[dict]):
            Historial conversacional previo.
    """
    session_id: str
    user_message: str
    lead_state: Optional[Dict[str, Any]] = None
    catalog_context: str = ""
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


# =========================
# ENDPOINTS
# =========================

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint simple para validar que la API está viva.

    Returns:
        dict: estado básico del servicio.
    """
    return {"status": "ok"}


@app.get("/lead-state/empty")
@app.get("/session/{session_id}")
def get_session_state(session_id: str) -> Dict[str, Any]:
    """
    Retorna el contenido actual de una sesión.
    """
    return get_session(session_id)

@app.get("/sessions/{session_id}/conversation")
def get_session_conversation(session_id: str):
    """
    Devuelve la conversación completa asociada a una sesión.

    La conversación se consulta desde:
    - memoria, si está activa,
    - JSON persistido, si el servidor fue reiniciado.
    """
    session = get_existing_session(session_id)

    if not session:
        return {
            "session_id": session_id,
            "found": False,
            "conversation_history": [],
            "message": "No se encontró historial para esta sesión.",
        }

    return {
        "session_id": session_id,
        "found": True,
        "conversation_history": session.get("conversation_history", []),
    }
    
@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    """
    Elimina una sesión específica.
    """
    clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.delete("/sessions")
def delete_all_sessions() -> Dict[str, str]:
    """
    Elimina todas las sesiones activas.
    """
    clear_all_sessions()
    return {"status": "all sessions cleared"}


def empty_lead_state() -> Dict[str, Any]:
    """
    Retorna un lead vacío con la estructura oficial del sistema.

    Útil para:
    - pruebas,
    - documentación,
    - inicialización de sesiones.

    Returns:
        dict: lead vacío serializado.
    """
    return LeadModel().model_dump()


@app.post("/premium-agent")
def premium_agent(payload: LeadAgentRequest) -> Dict[str, Any]:
    """
    Endpoint principal del agente comercial premium.

    Flujo:
    1. obtiene o crea la sesión,
    2. reutiliza el estado acumulado del lead,
    3. usa el historial persistido como fuente principal,
    4. procesa el mensaje,
    5. actualiza el historial completo,
    6. guarda la sesión.

    Importante:
    - conversation_history usado para persistencia debe ser el historial completo.
    - El frontend puede enviar historial, pero no debe sobrescribir el historial
      persistido si ya existe uno en la sesión.
    - Esto permite que "Ver conversación" muestre el historial completo del lead.
    """
    # -----------------------------
    # 1. Obtener o crear sesión
    # -----------------------------
    session = get_session(payload.session_id)

    # -----------------------------
    # 2. Resolver estado del lead
    # -----------------------------
    # Si el request trae lead_state, lo usamos.
    # Si no, usamos el estado acumulado en sesión.
    lead_state = (
        payload.lead_state
        if payload.lead_state is not None
        else session.get("lead_state")
    )

    # -----------------------------
    # 3. Resolver historial conversacional
    # -----------------------------
    # Para persistencia y auditoría usamos como fuente principal
    # el historial guardado en sesión.
    #
    # Solo usamos payload.conversation_history si la sesión aún no tiene historial.
    # Esto evita que un historial parcial enviado por la UI sobrescriba
    # la conversación completa.
    stored_history = session.get("conversation_history") or []

    if stored_history:
        conversation_history = stored_history
    else:
        conversation_history = payload.conversation_history or []

    # -----------------------------
    # 4. Procesar mensaje
    # -----------------------------
    result = process_lead_message(
        session_id=payload.session_id,
        user_message=payload.user_message,
        lead_state=lead_state,
        conversation_history=conversation_history,
        catalog_context=payload.catalog_context,
    )

    # -----------------------------
    # 5. Actualizar historial completo
    # -----------------------------
    updated_history = conversation_history.copy()

    updated_history.append({
        "role": "user",
        "content": payload.user_message,
    })

    updated_history.append({
        "role": "assistant",
        "content": result["assistant_reply"],
    })

    # -----------------------------
    # 6. Guardar sesión
    # -----------------------------
    # update_session guarda:
    # - lead_state en memoria,
    # - conversation_history en data/conversations.json.
    update_session(
        payload.session_id,
        result["updated_lead_state"],
        updated_history,
    )

    return result

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# API Leads
# -----------------------------
from pathlib import Path
import json

LEADS_FILE = Path("data/leads.json")

@app.get("/leads")
def get_leads():
    """
    Devuelve los leads guardados en data/leads.json.
    """
    if not LEADS_FILE.exists():
        return []

    with open(LEADS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# UI Leads
# -----------------------------
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

@app.get("/ui/leads", response_class=HTMLResponse)
def leads_ui(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="leads.html"
    )

@app.get("/ui/chat", response_class=HTMLResponse)
def chat_ui(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="chat.html"
    )


@app.get("/ui/widget", response_class=HTMLResponse)
def widget_ui(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="widgetVolvo.html"
    )