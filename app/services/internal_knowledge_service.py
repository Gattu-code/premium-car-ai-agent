"""
Módulo: internal_knowledge_service

Responsabilidad:
Construir contexto interno del negocio antes de llamar al modelo LLM.

Objetivo:
- Usar primero datos internos controlados.
- Reducir alucinaciones del modelo.
- Enriquecer el prompt con reglas comerciales y criterios de atención.
- Mantener separado el conocimiento de negocio del catálogo de productos.

Importante:
Este módulo NO debe consultar el catálogo.
La recuperación de productos pertenece a catalog_service.py.
"""

from typing import Any, Dict


def get_internal_context(
    user_message: str,
    lead_state: Dict[str, Any],
    brand: str = "volvo",
    market: str = "colombia",
) -> str:
    """
    Construye contexto interno relevante para el LLM.

    Responsabilidad:
    - Aportar reglas comerciales generales.
    - Aportar criterios de atención al lead.
    - Complementar el contexto del catálogo sin duplicarlo.

    Args:
        user_message (str): mensaje actual del usuario.
        lead_state (Dict[str, Any]): estado actual del lead.
        brand (str): marca activa del agente.
        market (str): mercado activo del agente.

    Returns:
        str: contexto interno listo para agregarse al prompt.
    """

    lead_has_phone = bool(lead_state.get("phone"))
    lead_has_email = bool(lead_state.get("email"))
    lead_has_city = bool(lead_state.get("city"))
    lead_has_budget = bool(lead_state.get("budget_range"))

    context_lines = [
        "DATOS INTERNOS DE NEGOCIO:",
        f"- Marca activa: {brand}.",
        f"- Mercado activo: {market}.",
        "- El agente debe usar únicamente el catálogo relevante recibido en el contexto.",
        "- No debe inventar modelos, precios, autonomías, versiones ni beneficios.",
        "- Si el catálogo no contiene información suficiente, debe aclararlo o hacer una pregunta breve.",
        "- Debe mantener tono consultivo, premium y natural.",
    ]

    if not lead_has_city:
        context_lines.append(
            "- Falta ciudad del cliente: pedirla de forma natural cuando sea oportuno."
        )

    if not lead_has_phone and not lead_has_email:
        context_lines.append(
            "- Falta contacto del cliente: solicitar teléfono y correo de forma natural, sin sonar a formulario."
        )

    if lead_has_phone:
        context_lines.append(
            "- El cliente ya entregó teléfono: está prohibido volver a pedirlo."
        )

    if lead_has_email:
        context_lines.append(
            "- El cliente ya entregó correo: está prohibido volver a pedirlo."
        )

    if not lead_has_budget:
        context_lines.append(
            "- Falta presupuesto: pedirlo antes de profundizar en versiones o cierre comercial."
        )

    if lead_state.get("lead_temperature") == "caliente":
        context_lines.append(
            "- Lead caliente: priorizar siguiente paso comercial, como prueba de manejo o contacto con asesor."
        )

    return "\n".join(context_lines)

def get_business_rules_list(
    brand: str = "volvo",
    market: str = "colombia",
) -> list[dict]:
    """
    Devuelve reglas comerciales estructuradas para RAG V3.

    Responsabilidad:
    - Reutilizar el conocimiento interno del negocio.
    - Entregar reglas estructuradas para embeddings.
    - Evitar crear archivos duplicados de reglas mientras el MVP usa lógica local.

    Args:
        brand (str): marca activa del agente.
        market (str): mercado activo del agente.

    Returns:
        list[dict]: reglas comerciales para indexar en RAG V3.
    """
    return [
        {
            "name": "telefono obligatorio para agendar",
            "description": "Para confirmar una prueba de manejo o visita, el teléfono del cliente es obligatorio. El correo no reemplaza el teléfono."
        },
        {
            "name": "no repetir datos ya capturados",
            "description": "Si el cliente ya entregó nombre, ciudad, teléfono, correo, presupuesto o modelo de interés, el agente no debe volver a pedirlo."
        },
        {
            "name": "financiacion",
            "description": "Si el cliente menciona financiación, financiar, crédito, cuotas, leasing o pago a plazos, registrar payment_method como financiacion."
        },
        {
            "name": "cierre de prueba de manejo",
            "description": "Si hay modelo, ciudad, teléfono, fecha y hora, el lead debe avanzar a schedule_test_drive."
        },
        {
            "name": "lead caliente",
            "description": "Un cliente con intención de prueba de manejo, teléfono y modelo de interés debe considerarse lead caliente."
        }
    ]