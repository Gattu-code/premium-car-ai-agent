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
    if appointment_is_complete(lead):
        return "schedule_test_drive"
    
    # Caso de lead muy fuerte
    if (
        lead.lead_temperature == "caliente"
        and lead.priority_score >= 50
        and not lead.phone
    ):
        return "request_contact_info"

    # Caso de lead listo para pasar a asesor
    if lead.is_ready_for_sales() and lead.phone:
        return "send_to_sales_advisor"

    # Caso de máxima prioridad
    if (
        lead.lead_temperature == "caliente"
        and lead.priority_score >= 80
        and lead.phone
    ):
        return "high_priority_lead"

    if current_action in [
        "continue_conversation",
        "request_contact_info",
        "send_to_sales_advisor",
        "high_priority_lead",
        "schedule_test_drive",
    ]:
        return current_action

    return "continue_conversation"