"""
Módulo: premium_agent

Responsabilidad:
Orquestar la interacción entre:
- el estado actual del lead,
- el prompt base del agente,
- el historial de conversación,
- el catálogo/contexto comercial,
- y el modelo de IA en ai_provider
OpenRouter/Gemini para generación y Ollama para embeddings

Este módulo:
1. construye el prompt final,
2. llama al modelo,
3. interpreta la respuesta JSON,
4. actualiza el estado del lead,
5. recalcula el estado comercial.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from app.services.ai_provider import generate_ai_response
from app.models.lead_model import LeadModel
from app.services.lead_normalizer import normalize_lead
from app.services.lead_router import decide_next_action
from app.storage.lead_store import save_lead
from app.services.vector_retrieval_service_v3 import retrieve_relevant_context_v3
from app.services.dealer_location_service import (
    get_dealers_list,
    build_dealer_context_for_prompt,
)
from app.services.brand_knowledge_service import get_brand_knowledge_list
from app.services.brand_knowledge_service import (
    build_brand_context_for_prompt,
)
from app.services.internal_knowledge_service import (
    get_internal_context,
    get_business_rules_list
)
from app.services.catalog_service import (
    load_catalog,
    build_catalog_context,
)
from app.utils.debug import debug_prompt

# Ruta del prompt base del agente
PROMPT_PATH = Path("app/prompts/premium_prompt.txt")

BRAND_CONTEXT_MODE = "structured"  # opciones: "structured" o "rag"
CATALOG_CONTEXT_MODE = "structured"
DEALER_CONTEXT_MODE = "structured"
RULES_CONTEXT_MODE = "structured"

def load_base_prompt() -> str:
    """
    Carga desde archivo el prompt base del agente.

    Returns:
        str: contenido del archivo premium_prompt.txt.
    """
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(
    user_message: str,
    lead_state: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    context: str = "",
) -> str:
    """
    Construye el prompt final enviado al LLM.

    Responsabilidad:
    - Combinar instrucciones base del agente.
    - Incluir contexto RAG V3 multi-fuente.
    - Incluir historial reciente y estado actual del lead.
    - Exigir salida en JSON válido.

    Contexto esperado:
    - catálogo de modelos recuperados,
    - conocimiento de marca,
    - concesionarios / sedes,
    - reglas internas de negocio.

    Args:
        user_message (str): último mensaje del cliente.
        lead_state (Dict[str, Any]): estado acumulado del lead.
        conversation_history (Optional[List[Dict[str, str]]]): historial reciente.
        context (str): contexto final construido por RAG V3.

    Returns:
        str: prompt completo listo para enviar al modelo.
    """
    base_prompt = load_base_prompt()

    history_text = json.dumps(
        conversation_history or [],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    lead_text = json.dumps(
        lead_state or {},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    prompt = f"""
{base_prompt}

----------------------------------------
CONTEXTO RAG V3 DISPONIBLE
----------------------------------------
{context if context else "No hay contexto adicional disponible."}

----------------------------------------
HISTORIAL RECIENTE
----------------------------------------
{history_text}

----------------------------------------
ESTADO ACTUAL DEL LEAD
----------------------------------------
{lead_text}

----------------------------------------
MENSAJE ACTUAL DEL CLIENTE
----------------------------------------
{user_message}

----------------------------------------
REGLAS FINALES DE SALIDA
----------------------------------------
- Responde únicamente en JSON válido.
- No incluyas texto antes ni después del JSON.
- Usa solo la información disponible en el contexto.
- No inventes modelos, sedes, direcciones, precios, autonomías ni beneficios.
- No preguntes datos que ya estén en el estado del lead.
- Haz máximo 1 pregunta útil en assistant_reply.
""".strip()

    return prompt

def safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Intenta convertir la respuesta del modelo en JSON.

    Estrategia:
    1. intenta parsear el texto completo,
    2. si falla, intenta extraer el bloque entre el primer '{' y el último '}'.

    Args:
        raw_text (str): texto bruto retornado por el modelo.

    Returns:
        dict: JSON interpretado.

    Raises:
        ValueError: si no se encuentra JSON válido.
    """
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start != -1 and end != -1 and end > start:
            candidate = raw_text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"No fue posible parsear el JSON del modelo: {exc}"
                ) from exc

        raise ValueError("La respuesta del modelo no contiene JSON válido.")

def sanitize_quick_replies(value, max_items: int = 8) -> list[str]:
    """
    Normaliza y valida las respuestas rápidas generadas por el LLM.

    Reglas:
    - Deben ser lista.
    - Máximo configurable de opciones.
    - Cada opción debe ser texto corto.
    - Se eliminan vacíos y duplicados.
    """
    if not isinstance(value, list):
        return []

    cleaned = []
    seen = set()

    for item in value:
        text = str(item or "").strip()

        if not text:
            continue

        if len(text) > 40:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

        if len(cleaned) >= max_items:
            break

    return cleaned

def normalize_agent_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Garantiza que existan las claves mínimas esperadas.

    Si el modelo omite alguna clave, se completa con valores por defecto.
    Si el modelo devuelve quick_replies, se conservan para sanitizarlas después.
    """
    return {
        "assistant_reply": parsed.get("assistant_reply", ""),
        "quick_replies": parsed.get("quick_replies", []),
        "updated_lead_state": parsed.get("updated_lead_state", {}),
        "next_action": parsed.get("next_action", "continue_conversation"),
        "confidence": parsed.get("confidence", 0.0),
    }


def process_lead_message(
    session_id: str,
    user_message: str,
    lead_state: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    catalog_context: str = "",
) -> Dict[str, Any]:
    """
    Procesa un mensaje del usuario y devuelve la respuesta estructurada del agente AI.

    Flujo completo:
    1. Inicializa el estado del lead si no existe.
    2. Carga el catálogo activo y construye contexto de productos.
    3. Construye el prompt contextual para el modelo.
    4. Ejecuta la inferencia en el modelo (Ollama).
    5. Parsea y normaliza la respuesta del modelo.
    6. Actualiza el estado del LeadModel con la información inferida.
    7. Aplica reglas de negocio (normalización y enriquecimiento).
    8. Recalcula el estado comercial del lead.
    9. Decide la siguiente acción (lógica backend).
    10. Ajusta la respuesta si se requiere solicitar contacto.
    11. Persiste el lead si está en etapa de contacto.

    Args:
        user_message (str): mensaje actual del usuario.
        lead_state (dict, optional): estado previo del lead.
        conversation_history (list[dict], optional): historial previo.
        catalog_context (str): parámetro reservado para contexto de catálogo.
            Se sobrescribe dinámicamente con el catálogo activo.

    Returns:
        dict: respuesta estructurada del agente con:
            - assistant_reply (str)
            - updated_lead_state (dict)
            - next_action (str)
            - confidence (float)
    """
    # -----------------------------
    # 1. Inicialización del estado
    # -----------------------------
    lead_state = lead_state or LeadModel().model_dump()


    # -----------------------------
    # 2. Construcción de contexto (Structured + RAG-ready)
    # -----------------------------
    # Responsabilidad:
    # Preparar el contexto que será entregado al LLM combinando:
    # - catálogo de productos (vehículos disponibles)
    # - conocimiento de marca (seguridad, tecnología, etc.)
    # - concesionarios / sedes disponibles
    # - contexto interno del negocio
    # - reglas internas recuperadas cuando aplique
    #
    # Principio de diseño:
    # - El backend NO decide qué vehículo recomendar.
    # - El backend SOLO prepara fuentes confiables de contexto.
    # - Para datos pequeños y críticos usamos contexto estructurado completo.
    # - Para conocimiento amplio o creciente podemos usar RAG con embeddings.
    # - El LLM razona y genera la recomendación usando ese contexto.
    #
    # Esto permite:
    # - escalar a múltiples marcas
    # - evitar lógica rígida en código
    # - reducir alucinaciones en datos críticos
    # - evolucionar fácilmente a RAG con embeddings cuando el volumen crezca

    catalog = load_catalog(
    brand="volvo",
    market="colombia",
    )

    dealers = get_dealers_list(
        brand="volvo",
        market="colombia",
    )

    brand_knowledge = get_brand_knowledge_list(
        brand="volvo",
        market="colombia",
    )

    business_rules = get_business_rules_list(
        brand="volvo",
        market="colombia",
    )

    # -----------------------------
    # 2.2 Contexto RAG opcional según switches
    # -----------------------------
    # El RAG solo se ejecuta si alguna fuente está configurada en modo "rag".
    # Para el MVP, catálogo, dealers y brand knowledge pueden ir estructurados.
    # Esto evita depender de embeddings para datos pequeños y críticos.

    needs_rag = (
        CATALOG_CONTEXT_MODE == "rag"
        or DEALER_CONTEXT_MODE == "rag"
        or BRAND_CONTEXT_MODE == "rag"
        or RULES_CONTEXT_MODE == "rag"
    )

    if needs_rag:
        rag_context = retrieve_relevant_context_v3(
            user_message=user_message,
            lead_state=lead_state,
            catalog=catalog,
            dealers=dealers,
            brand_knowledge=brand_knowledge,
            business_rules=business_rules,
        )
    else:
        rag_context = {
            "models": [],
            "dealers": [],
            "brand_context": "",
            "rules_context": "",
        }

    models = rag_context.get("models", [])
    dealer_context_data = rag_context.get("dealers", [])

    # -----------------------------
    # DEBUG TEMPORAL
    # -----------------------------
    if needs_rag:
        print("USANDO VECTOR RETRIEVAL V3")
        print("MODELOS RECUPERADOS:", [m.get("model") for m in models])
        print("DEALERS RECUPERADOS:", [d.get("name") for d in dealer_context_data])
    else:
        print("RAG V3 DESACTIVADO PARA ESTA RESPUESTA")
        print("USANDO CONTEXTO ESTRUCTURADO")

    # -----------------------------
    # 2.3 Contexto de catálogo
    # -----------------------------
    # structured: entrega todo el catálogo activo.
    # rag: entrega solo los modelos recuperados por embeddings.
    if CATALOG_CONTEXT_MODE == "structured":
        catalog_context = build_catalog_context(catalog)
    else:
        catalog_context = build_catalog_context(models)

    # -----------------------------
    # 2.4 Contexto interno de negocio
    # -----------------------------
    internal_context = get_internal_context(
        user_message=user_message,
        lead_state=lead_state,
        brand="volvo",
        market="colombia",
    )

    # -----------------------------
    # 2.5 Contexto de conocimiento de marca
    # -----------------------------
    # structured: entrega todo el conocimiento de marca cargado.
    # rag: entrega solo temas recuperados por embeddings.
    if BRAND_CONTEXT_MODE == "structured":
        brand_context = build_brand_context_for_prompt(
            brand_knowledge=brand_knowledge,
        )
    else:
        brand_context = rag_context.get("brand_context", "")

    # -----------------------------
    # 2.6 Contexto de concesionarios / sedes
    # -----------------------------
    # structured: entrega dealers desde JSON completo, priorizando ciudad del lead.
    # rag: entrega solo dealers recuperados por embeddings.
    if DEALER_CONTEXT_MODE == "structured":
        dealer_context = build_dealer_context_for_prompt(
            dealers=dealers,
            lead_state=lead_state,
            user_message=user_message,
        )
    else:
        dealer_context = build_dealer_context_for_prompt(
            dealers=dealer_context_data,
            lead_state=lead_state,
            user_message=user_message,
        )
    # -----------------------------
    # 2.7 Contexto de reglas internas adicionales
    # -----------------------------
    # Por ahora, si RULES_CONTEXT_MODE = "rag", usamos rules_context del RAG.
    # Si está en "structured", dejamos este bloque vacío porque ya tenemos
    # get_internal_context() como fuente determinística principal.
    if RULES_CONTEXT_MODE == "rag":
        rules_context = rag_context.get("rules_context", "")
    else:
        rules_context = ""

    # -----------------------------
    # 2.8 Fecha actual del sistema
    # -----------------------------
    # Entregamos al LLM la fecha actual para interpretar referencias como:
    # "mañana", "próximo martes", "este sábado".
    # La normalización final sigue siendo responsabilidad del backend.
    weekday_labels = [
        "lunes",
        "martes",
        "miércoles",
        "jueves",
        "viernes",
        "sábado",
        "domingo",
    ]

    now_bogota = datetime.now(ZoneInfo("America/Bogota"))
    weekday_name = weekday_labels[now_bogota.weekday()]

    current_date_context = (
        f"Fecha actual del sistema: {now_bogota.strftime('%Y-%m-%d')}.\n"
        f"Día de la semana: {weekday_name}.\n"
        "Zona horaria: America/Bogota.\n"
        "Usa esta fecha como referencia para entender expresiones como "
        "'mañana', 'este sábado' o 'próximo martes'. "
        "La fecha normalizada final la calcula el backend."
    )



    # -----------------------------
    # 2.8 Contexto final enviado al LLM
    # -----------------------------
    final_context = f"""
FECHA ACTUAL:
{current_date_context}

CATÁLOGO RELEVANTE:
{catalog_context}

CONOCIMIENTO DE MARCA:
{brand_context if brand_context else "No aplica para esta conversación."}

CONCESIONARIOS / SEDES:
{dealer_context if dealer_context else "No hay contexto de sedes disponible. No inventar sedes, direcciones, entregas ni disponibilidad local."}

CONTEXTO DE NEGOCIO:
{internal_context}

REGLAS INTERNAS RECUPERADAS:
{rules_context if rules_context else "No aplica para esta conversación."}
""".strip()

   
    # -----------------------------
    # 3. Construcción del prompt final
    # -----------------------------
    # Limitamos el historial reciente para controlar consumo de tokens
    # y mantener el foco conversacional.
    conversation_history = (conversation_history or [])[-4:]

    prompt = build_prompt(
        user_message=user_message,
        lead_state=lead_state,
        conversation_history=conversation_history,
        context=final_context,
    )

    # -----------------------------
    # 4. Llamada al modelo (LLM)
    # -----------------------------
    debug_prompt(prompt)
    raw_response = generate_ai_response(prompt)

    # -----------------------------
    # 5. Parseo y normalización
    # -----------------------------
    parsed = safe_parse_json(raw_response)
    parsed = normalize_agent_response(parsed)

    # -----------------------------
    # 6. Actualización del LeadModel
    # -----------------------------
    updated_state = parsed.get("updated_lead_state", {})

    lead = LeadModel(**lead_state)
    lead.update_from_dict(updated_state)

    # Primera evaluación del estado comercial
    lead.refresh_business_state()

    # -----------------------------
    # 7. Normalización de negocio
    # -----------------------------
    lead = normalize_lead(lead)

    # -----------------------------
    # 8. Recalcular estado final
    # -----------------------------
    lead.refresh_business_state()

    # Sobrescribir el estado con el resultado final del backend
    parsed["updated_lead_state"] = lead.model_dump()
    
    # -----------------------------
    # Quick replies
    # -----------------------------
    # El LLM puede sugerir respuestas rápidas contextuales.
    # El backend las sanitiza para evitar ruido, duplicados o textos largos.
    max_quick_replies = 6

    assistant_reply_text = (parsed.get("assistant_reply") or "").lower()
    next_action = parsed.get("next_action") or ""

    if (
        next_action in [
            "dealer_city_selection_required",
            "dealer_selection_required",
        ]
        or "ciudades con vitrinas" in assistant_reply_text
        or "ciudades con sede" in assistant_reply_text
        or "vitrinas volvo disponibles" in assistant_reply_text
    ):
        max_quick_replies = 8

    parsed["quick_replies"] = sanitize_quick_replies(
        parsed.get("quick_replies", []),
        max_items=max_quick_replies,
    )

    # -----------------------------
    # 9. Decisión de siguiente acción
    # -----------------------------
    parsed["next_action"] = decide_next_action(
        lead=lead,
        current_action=parsed.get("next_action", "continue_conversation"),
    )

    # -----------------------------
    # 10. Ajuste de respuesta comercial
    # -----------------------------
    # Si el backend detecta que falta información de contacto,
    # mantenemos la intención comercial sin sonar genérico ni robótico.
    missing_contact_fields = []

    if not lead.phone:
        missing_contact_fields.append("teléfono")

    if not lead.email:
        missing_contact_fields.append("correo electrónico")

    if parsed.get("next_action") == "request_contact_info" and missing_contact_fields:
        if len(missing_contact_fields) == 2:
            contact_request = "tu número de teléfono y correo electrónico"
        else:
            contact_request = f"tu {missing_contact_fields[0]}"

        parsed["assistant_reply"] = (
            f"Perfecto, {lead.lead_name or ''}. Para avanzar con la visita, "
            f"¿me compartes {contact_request}?"
        ).replace("  ", " ").strip()

        parsed["quick_replies"] = []
        parsed.setdefault("updated_lead_state", lead.model_dump())
        parsed["updated_lead_state"]["pending_questions"] = []

    # -----------------------------
    # Dealer obligatorio antes de confirmar cita
    # -----------------------------
    # Si el cliente ya tiene intención de visita y fecha/hora,
    # pero no hay sede/vitrina específica, no confirmamos la cita.
    # Primero pedimos seleccionar una sede válida.
    appointment_location = lead.appointment_location or {}

    has_dealer = (
        isinstance(appointment_location, dict)
        and appointment_location.get("dealer_name")
        and appointment_location.get("city")
    )

    if (
        lead.interest_type == "visita"
        and lead.appointment_date
        and lead.appointment_time
        and not has_dealer
    ):
        city = (lead.city or "").strip()

        dealer_names = []

        for dealer in dealers:
            dealer_city = str(dealer.get("city", "")).strip()

            if city and dealer_city.lower() == city.lower():
                dealer_name = dealer.get("name") or dealer.get("dealer_name")

                if dealer_name and dealer_name not in dealer_names:
                    dealer_names.append(dealer_name)

        if dealer_names:
            parsed["assistant_reply"] = (
                f"Perfecto, {lead.lead_name or ''}. Antes de confirmar tu visita, "
                f"¿en cuál sede te gustaría conocer el vehículo en {city}?"
            ).replace("  ", " ").strip()

            parsed["quick_replies"] = dealer_names[:6]
            parsed["next_action"] = "dealer_selection_required"
            parsed.setdefault("updated_lead_state", lead.model_dump())
            parsed["updated_lead_state"]["pending_questions"] = []

    # -----------------------------
    # 11. Persistencia del lead
    # -----------------------------
    # Guardamos cada interacción para mantener trazabilidad por conversación.

    # -----------------------------
    # Limpieza de preguntas pendientes
    # -----------------------------
    # La respuesta visible ya va en assistant_reply.
    # Evitamos que la UI muestre una segunda pregunta duplicada.
    parsed.setdefault("updated_lead_state", lead.model_dump())
    parsed["updated_lead_state"]["pending_questions"] = []

    save_lead(session_id, parsed["updated_lead_state"])

    # -----------------------------
    # 12. Retorno final
    # -----------------------------
    print("PENDING QUESTIONS FINAL:", parsed["updated_lead_state"].get("pending_questions"))
    return parsed