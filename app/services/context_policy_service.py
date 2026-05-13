"""
Módulo: context_policy_service

Responsabilidad:
Decidir qué contexto enviar al LLM según el mensaje actual y el estado del lead.

Objetivo:
Reducir tokens evitando enviar siempre:
- catálogo completo
- conocimiento de marca completo
- sedes completas

Este servicio NO construye el prompt.
Este servicio SOLO decide la política de contexto.
"""

from typing import Any, Dict, List
import re

from app.services.date_normalizer import normalize_text


def appointment_is_complete_state(lead_state: Dict[str, Any]) -> bool:
    """
    Determina si el lead ya tiene los datos mínimos para considerar
    una cita completa.
    """
    appointment_location = lead_state.get("appointment_location") or {}

    return all(
        [
            lead_state.get("phone"),
            lead_state.get("email"),
            lead_state.get("city"),
            lead_state.get("vehicle_interest"),
            lead_state.get("appointment_date"),
            lead_state.get("appointment_time"),
            isinstance(appointment_location, dict),
            appointment_location.get("dealer_name"),
            appointment_location.get("address"),
            appointment_location.get("city"),
        ]
    )

# funciones de detección de intención y temas en el mensaje del usuario.
# estas funciones ayudan a decidir qué contexto es relevante enviar al LLM,
# evitando enviar siempre todo el catálogo, conocimiento de marca y sedes.  
def text_has_any(text: str, terms: List[str]) -> bool:
    """
    Busca términos completos en el texto normalizado.

    Evita falsos positivos como:
    - "no" dentro de "autonomia"
    - "ver" dentro de otra palabra más larga
    """
    normalized = normalize_text(text)

    for term in terms:
        normalized_term = normalize_text(term)

        if not normalized_term:
            continue

        pattern = r"\b" + re.escape(normalized_term) + r"\b"

        if re.search(pattern, normalized):
            return True

    return False

def is_closing_message(user_message: str) -> bool:
    return text_has_any(
        user_message,
        [
            "gracias",
            "muchas gracias",
            "ok",
            "listo",
            "perfecto",
            "vale",
            "entendido",
            "no",
            "nada mas",
            "hasta luego",
        ],
    )

def is_schedule_intent(
    user_message: str,
    lead_state: Dict[str, Any],
) -> bool:
    """
    Detecta si el mensaje actual requiere contexto de sedes/concesionarios.

    Importante:
    - Si el mensaje actual habla de visita/agendamiento, sí necesita dealers.
    - Si el lead venía en flujo de visita pero ahora pregunta algo técnico,
      no necesitamos volver a enviar dealers.
    - Si falta sede y el cliente probablemente está respondiendo ciudad/sede,
      sí necesitamos dealers.
    """

    current_message_has_schedule_signal = text_has_any(
        user_message,
        [
            "visita",
            "visitar",
            "verlo",
            "ver",
            "agendar",
            "agenda",
            "cita",
            "probar",
            "test drive",
            "manejo",
            "sede",
            "vitrina",
            "concesionario",
        ],
    )

    if current_message_has_schedule_signal:
        return True

    if is_brand_or_technical_question(user_message):
        return False

    appointment_location = lead_state.get("appointment_location") or {}

    has_dealer = (
        isinstance(appointment_location, dict)
        and appointment_location.get("dealer_name")
        and appointment_location.get("city")
    )

    if lead_state.get("interest_type") == "visita" and not has_dealer:
        return True

    return False


def is_brand_or_technical_question(user_message: str) -> bool:
    return text_has_any(
        user_message,
        [
            "seguridad",
            "seguro",
            "proteccion",
            "protección",
            "bebe",
            "bebé",
            "ninos",
            "niños",
            "familia",
            "bateria",
            "batería",
            "carga",
            "cargar",
            "autonomia",
            "autonomía",
            "electrico",
            "eléctrico",
            "hibrido",
            "híbrido",
            "tecnologia",
            "tecnología",
            "pico y placa",
            "beneficio",
            "garantia",
            "garantía",
        ],
    )


def asks_for_options_or_comparison(user_message: str) -> bool:
    return text_has_any(
        user_message,
        [
            "opciones",
            "modelos",
            "comparar",
            "comparacion",
            "comparación",
            "recomienda",
            "recomendacion",
            "recomendación",
            "cual me sirve",
            "cuál me sirve",
            "cual me recomiendas",
            "cuál me recomiendas",
        ],
    )


def select_relevant_catalog_models(
    catalog: List[Dict[str, Any]],
    user_message: str,
    lead_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Selecciona modelos relevantes del catálogo según el mensaje actual
    y el vehículo de interés ya capturado.
    """
    query = normalize_text(
        " ".join(
            [
                user_message or "",
                str(lead_state.get("vehicle_interest", "")),
            ]
        )
    )

    selected = []

    for model in catalog or []:
        model_name = str(model.get("model", "")).strip()

        if not model_name:
            continue

        model_name_normalized = normalize_text(model_name)
        model_short_name = normalize_text(model_name.split()[0])

        if model_name_normalized in query or model_short_name in query:
            selected.append(model)

    return selected


def build_context_policy(
    user_message: str,
    lead_state: Dict[str, Any],
    catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Decide qué contexto enviar al LLM.

    catalog_mode:
    - none: no enviar catálogo
    - selected: enviar solo modelo relevante
    - all: enviar catálogo completo

    include_brand:
    - True si la pregunta toca marca, tecnología, seguridad o electrificación

    include_dealers:
    - True si hay intención de visita, sede o agendamiento
    """
    appointment_complete = appointment_is_complete_state(lead_state)

    if appointment_complete and is_closing_message(user_message):
        return {
            "catalog_mode": "none",
            "include_brand": False,
            "include_dealers": False,
            "reason": "closing_after_confirmed_appointment",
        }

    schedule_intent = is_schedule_intent(
        user_message=user_message,
        lead_state=lead_state,
    )

    brand_question = is_brand_or_technical_question(user_message)
    asks_options = asks_for_options_or_comparison(user_message)

    selected_models = select_relevant_catalog_models(
        catalog=catalog,
        user_message=user_message,
        lead_state=lead_state,
    )

    if asks_options:
        catalog_mode = "all"
    elif selected_models or lead_state.get("vehicle_interest"):
        catalog_mode = "selected"
    elif schedule_intent:
        catalog_mode = "selected"
    else:
        catalog_mode = "none"

    return {
        "catalog_mode": catalog_mode,
        "include_brand": brand_question,
        "include_dealers": schedule_intent,
        "reason": "dynamic_context_policy",
    }