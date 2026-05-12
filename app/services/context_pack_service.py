"""
Módulo: context_pack_service

Responsabilidad:
Construir un paquete de contexto estructurado antes de llamar al LLM.

Objetivo:
- Darle al modelo una ficha clara del estado actual del lead.
- Reducir alucinaciones.
- Evitar repetir preguntas.
- Identificar qué datos faltan.
- Explicitar restricciones del backend antes de generar respuesta.

Importante:
Este módulo NO genera respuestas.
Este módulo NO recomienda vehículos.
Este módulo NO reemplaza el RAG.
Este módulo organiza el contexto para que el LLM converse mejor.
"""

from typing import Any, Dict, List


def field_has_value(value: Any) -> bool:
    """
    Evalúa si un campo tiene un valor útil.

    Args:
        value (Any): valor del campo.

    Returns:
        bool: True si el campo contiene información real.
    """
    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() not in ["", "null", "none", "-", "empty"]

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, dict):
        return len(value) > 0

    return bool(value)


def detect_missing_fields(lead_state: Dict[str, Any]) -> List[str]:
    """
    Detecta campos importantes faltantes para avanzar comercialmente.

    Nota:
    Teléfono y correo son obligatorios para confirmar una cita.

    Args:
        lead_state (Dict[str, Any]): estado actual del lead.

    Returns:
        List[str]: campos faltantes.
    """
    required_fields = [
        "lead_name",
        "city",
        "vehicle_interest",
        "phone",
        "email",
        "appointment_date",
        "appointment_time",
    ]

    missing = []

    for field in required_fields:
        if not field_has_value(lead_state.get(field)):
            missing.append(field)

    return missing


def detect_current_stage(
    lead_state: Dict[str, Any],
    missing_fields: List[str],
) -> str:
    """
    Detecta la etapa comercial actual del lead.

    Args:
        lead_state (Dict[str, Any]): estado actual del lead.
        missing_fields (List[str]): campos faltantes.

    Returns:
        str: etapa comercial detectada.
    """
    has_model = field_has_value(lead_state.get("vehicle_interest"))
    has_phone = field_has_value(lead_state.get("phone"))
    has_email = field_has_value(lead_state.get("email"))
    has_date = field_has_value(lead_state.get("appointment_date"))
    has_time = field_has_value(lead_state.get("appointment_time"))
    has_family_context = field_has_value(lead_state.get("family_context"))
    has_motivation = field_has_value(lead_state.get("primary_motivation"))

    if has_model and has_phone and has_email and has_date and has_time:
        return "appointment_ready_or_confirmed"

    if has_model and (has_date or has_time) and (not has_phone or not has_email):
        return "appointment_pending_contact"

    if has_model and has_phone and has_email and (not has_date or not has_time):
        return "appointment_pending_datetime"

    if has_model:
        return "model_selected"

    if has_family_context or has_motivation:
        return "discovery"

    return "initial_capture"


def detect_next_best_action(
    stage: str,
    missing_fields: List[str],
    dealer_context_data: List[Dict[str, Any]],
) -> str:
    """
    Sugiere la siguiente mejor acción conversacional.

    Importante:
    Esta función no genera la respuesta final.
    Solo orienta al LLM sobre el siguiente paso lógico.

    Args:
        stage (str): etapa comercial actual.
        missing_fields (List[str]): campos faltantes.
        dealer_context_data (List[Dict[str, Any]]): sedes recuperadas.

    Returns:
        str: siguiente acción sugerida.
    """
    if stage == "initial_capture":
        if "lead_name" in missing_fields and "city" in missing_fields:
            return "pedir nombre y ciudad"
        if "lead_name" in missing_fields:
            return "pedir nombre"
        if "city" in missing_fields:
            return "pedir ciudad"

    if stage == "discovery":
        return "entender mejor la necesidad antes de recomendar"

    if stage == "model_selected":
        return "resolver dudas del modelo o avanzar a visita/prueba"

    if stage == "appointment_pending_contact":
        if "phone" in missing_fields and "email" in missing_fields:
            return "pedir teléfono y correo"
        if "phone" in missing_fields:
            return "pedir teléfono"
        if "email" in missing_fields:
            return "pedir correo"

    if stage == "appointment_pending_datetime":
        if "appointment_date" in missing_fields and "appointment_time" in missing_fields:
            return "pedir fecha y hora"
        if "appointment_date" in missing_fields:
            return "pedir fecha"
        if "appointment_time" in missing_fields:
            return "pedir hora"

    if stage == "appointment_ready_or_confirmed":
        if dealer_context_data:
            return "confirmar cita con sede disponible"
        return "no confirmar sede; validar ubicación con asesor"

    return "continuar conversación"


def build_context_pack(
    user_message: str,
    lead_state: Dict[str, Any],
    models: List[Dict[str, Any]],
    dealer_context_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Construye el Context Pack para el LLM.

    Responsabilidad:
    - Resumir datos confirmados del cliente.
    - Identificar campos faltantes.
    - Indicar etapa comercial.
    - Explicitar restricciones.
    - Sugerir la siguiente mejor acción.

    Args:
        user_message (str): mensaje actual del cliente.
        lead_state (Dict[str, Any]): estado acumulado del lead.
        models (List[Dict[str, Any]]): modelos recuperados por RAG.
        dealer_context_data (List[Dict[str, Any]]): dealers recuperados por RAG.

    Returns:
        Dict[str, Any]: paquete estructurado y texto listo para prompt.
    """
    missing_fields = detect_missing_fields(lead_state)

    stage = detect_current_stage(
        lead_state=lead_state,
        missing_fields=missing_fields,
    )

    next_best_action = detect_next_best_action(
        stage=stage,
        missing_fields=missing_fields,
        dealer_context_data=dealer_context_data,
    )

    confirmed_fields = {
        "lead_name": lead_state.get("lead_name", ""),
        "city": lead_state.get("city", ""),
        "vehicle_interest": lead_state.get("vehicle_interest", ""),
        "phone": lead_state.get("phone", ""),
        "email": lead_state.get("email", ""),
        "appointment_date": lead_state.get("appointment_date", ""),
        "appointment_time": lead_state.get("appointment_time", ""),
        "family_context": lead_state.get("family_context", ""),
        "primary_motivation": lead_state.get("primary_motivation", ""),
        "payment_method": lead_state.get("payment_method", ""),
    }

    available_models = [
        model.get("model", "")
        for model in models
        if model.get("model")
    ]

    available_dealers = [
        {
            "dealer_name": dealer.get("name", ""),
            "city": dealer.get("city", ""),
            "address": dealer.get("address")
            or dealer.get("sales_address")
            or "",
        }
        for dealer in dealer_context_data
    ]

    restrictions = []

    if not dealer_context_data and field_has_value(lead_state.get("city")):
        restrictions.append(
            "No hay sede/concesionario recuperado para la ciudad actual. No inventar sede, dirección, entrega local ni disponibilidad."
        )

    if "phone" in missing_fields:
        restrictions.append("No confirmar cita sin teléfono.")

    if "email" in missing_fields:
        restrictions.append("No confirmar cita sin correo electrónico.")

    if "appointment_date" in missing_fields or "appointment_time" in missing_fields:
        restrictions.append("No confirmar cita sin fecha y hora.")

    context_text = f"""
CONTEXT PACK

Mensaje actual:
{user_message}

Datos confirmados del cliente:
- Nombre: {confirmed_fields.get("lead_name") or "No confirmado"}
- Ciudad: {confirmed_fields.get("city") or "No confirmada"}
- Modelo/interés: {confirmed_fields.get("vehicle_interest") or "No confirmado"}
- Teléfono: {confirmed_fields.get("phone") or "No confirmado"}
- Correo: {confirmed_fields.get("email") or "No confirmado"}
- Fecha: {confirmed_fields.get("appointment_date") or "No confirmada"}
- Hora: {confirmed_fields.get("appointment_time") or "No confirmada"}
- Contexto familiar: {confirmed_fields.get("family_context") or "No confirmado"}
- Motivación principal: {confirmed_fields.get("primary_motivation") or "No confirmada"}
- Forma de pago: {confirmed_fields.get("payment_method") or "No confirmada"}

Etapa comercial actual:
- {stage}

Campos faltantes:
- {", ".join(missing_fields) if missing_fields else "Ninguno"}

Modelos disponibles en contexto:
- {", ".join(available_models) if available_models else "Ninguno"}

Sedes disponibles en contexto:
- {available_dealers if available_dealers else "Ninguna"}

Restricciones:
- {" | ".join(restrictions) if restrictions else "Sin restricciones críticas adicionales"}

Siguiente mejor acción:
- {next_best_action}
""".strip()

    return {
        "confirmed_fields": confirmed_fields,
        "missing_fields": missing_fields,
        "stage": stage,
        "available_models": available_models,
        "available_dealers": available_dealers,
        "restrictions": restrictions,
        "next_best_action": next_best_action,
        "text": context_text,
    }