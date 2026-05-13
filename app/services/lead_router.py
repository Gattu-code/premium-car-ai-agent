"""
Módulo: lead_router

Responsabilidad:
Definir la acción comercial recomendada para un lead
según su estado actual.

Este módulo actúa como una capa de decisión backend
para no depender únicamente de la recomendación del modelo.
"""

from app.models.lead_model import LeadModel

# Función principal de este módulo: decide la siguiente acción comercial recomendada
# basada en reglas de negocio aplicadas sobre el estado actual del lead.
# Estas reglas pueden sobrescribir la recomendación del modelo si detectan casos claros.
# Por ejemplo:
# - Si el lead está muy calificado pero no tiene teléfono, es prioritario solicitar contacto.
# - Si el lead está listo para asesor pero no tiene teléfono, es prioritario solicitar contacto     
# - Si el lead está listo para asesor y tiene teléfono, es prioritario enviarlo a asesor.   
def appointment_is_complete(lead) -> bool:
    """
    Determina si el lead ya tiene todos los datos mínimos
    para considerar una cita confirmada.
    """
    appointment_location = lead.appointment_location or {}

    return all(
         [
            lead.lead_name,
            lead.phone,
            lead.email,
            lead.city,
            lead.vehicle_interest,
            lead.appointment_date,
            lead.appointment_time,
            isinstance(appointment_location, dict),
            appointment_location.get("dealer_name"),
            appointment_location.get("address"),
            appointment_location.get("city"),
        ]
    )
# Función para detectar si la cita está casi completa pero falta solo el nombre del cliente.
# Esto es útil para priorizar la solicitud del nombre sin sonar genérico, cuando ya tenemos casi toda la información para agendar la cita.
# Si detectamos este caso, la acción recomendada será "request_lead_name" para solicitar solo el nombre del cliente, en lugar de una solicitud genérica de información de contacto.
# Esta función se usa en decide_next_action() para mejorar la personalización de la interacción en esta etapa crítica del proceso comercial.
# La función appointment_is_complete() sigue considerando la cita como completa solo si ya tenemos el nombre del cliente, pero appointment_missing_only_name() nos ayuda a identificar ese caso específico donde solo falta el nombre, permitiendo una acción comercial más precisa.
# Por ejemplo, si el lead ha proporcionado teléfono, email, ciudad, interés en vehículo, fecha y hora de cita, y detalles de la ubicación de la cita, pero no ha proporcionado su nombre, esta función detectará ese caso y la acción recomendada será solicitar solo el nombre del cliente para completar la información necesaria para agendar la cita.
# Esto mejora la experiencia del cliente al hacer una solicitud más específica y relevante en lugar de pedir información de contacto de manera genérica cuando ya tenemos casi toda la información necesaria para avanzar hacia el cierre de la cita.
# Esta función se integra en la lógica de decide_next_action() para asegurar que estamos priorizando la solicitud de información de manera inteligente según el estado actual del lead.

def appointment_missing_only_name(lead) -> bool:
    """
    Detecta si la cita está casi completa, pero falta el nombre del cliente.
    """
    appointment_location = lead.appointment_location or {}

    has_appointment_data = all(
        [
            lead.phone,
            lead.email,
            lead.city,
            lead.vehicle_interest,
            lead.appointment_date,
            lead.appointment_time,
            isinstance(appointment_location, dict),
            appointment_location.get("dealer_name"),
            appointment_location.get("address"),
            appointment_location.get("city"),
        ]
    )

    return has_appointment_data and not lead.lead_name

# Reglas de negocio para decidir la acción comercial recomendada.
# Estas reglas se aplican sobre el estado actual del lead y pueden
# sobrescribir la recomendación del modelo si detectan casos claros.
# Por ejemplo:
# - Si el lead está muy calificado pero no tiene teléfono, es prioritario solicitar contacto.
# - Si el lead está listo para asesor pero no tiene teléfono, es prioritario solicitar contacto.
# - Si el lead está listo para asesor y tiene teléfono, es prioritario enviarlo a asesor.  

def decide_next_action(lead: LeadModel, current_action: str = "continue_conversation") -> str:
    """
    Decide la siguiente acción comercial para el lead.

    Reglas:
    - Si la cita está casi completa pero falta nombre:
      request_lead_name
    - Si la cita está completa:
      schedule_test_drive
    - Si el lead está muy calificado pero no tiene teléfono:
      request_contact_info
    - Si el lead está calificado y tiene teléfono:
      send_to_sales_advisor
    - Si el lead tiene alta prioridad:
      high_priority_lead
    - En otros casos:
      continue_conversation
    """

    if appointment_missing_only_name(lead):
        return "request_lead_name"

    if appointment_is_complete(lead):
        return "schedule_test_drive"

    if (
        lead.lead_temperature == "caliente"
        and lead.priority_score >= 50
        and not lead.phone
    ):
        return "request_contact_info"

    if lead.is_ready_for_sales() and lead.phone:
        return "send_to_sales_advisor"

    if (
        lead.lead_temperature == "caliente"
        and lead.priority_score >= 80
        and lead.phone
    ):
        return "high_priority_lead"

    if current_action in [
        "continue_conversation",
        "request_contact_info",
        "request_lead_name",
        "send_to_sales_advisor",
        "high_priority_lead",
        "schedule_test_drive",
    ]:
        return current_action

    return "continue_conversation"