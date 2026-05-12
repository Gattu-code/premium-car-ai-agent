"""
Módulo: lead_router

Responsabilidad:
Definir la acción comercial recomendada para un lead
según su estado actual.

Este módulo actúa como una capa de decisión backend
para no depender únicamente de la recomendación del modelo.
"""

from app.models.lead_model import LeadModel


def decide_next_action(lead: LeadModel, current_action: str = "continue_conversation") -> str:
    """
    Decide la siguiente acción comercial para el lead.

    Reglas:
    - Si el lead está muy calificado pero no tiene teléfono:
      request_contact_info
    - Si el lead está calificado y tiene teléfono:
      send_to_sales_advisor
    - Si el lead tiene alta prioridad:
      high_priority_lead
    - En otros casos:
      continue_conversation

    Args:
        lead (LeadModel): estado actual del lead
        current_action (str): acción sugerida por la IA

    Returns:
        str: acción final recomendada
    """

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

    # Si el modelo ya sugirió algo útil, se puede respetar solo si no contradice reglas
    if current_action in [
        "continue_conversation",
        "request_contact_info",
        "send_to_sales_advisor",
        "high_priority_lead",
    ]:
        return current_action

    return "continue_conversation"