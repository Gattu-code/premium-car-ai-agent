"""
Módulo: dealer_location_service

Responsabilidad:
Recuperar contexto de vitrinas, talleres y concesionarios disponibles
para una marca y mercado.

Objetivo:
- Separar la información de sedes/concesionarios del catálogo de productos.
- Permitir que el agente oriente al cliente según su ciudad.
- Preparar la arquitectura para multi-marca y multi-cliente.
- Evitar quemar direcciones o teléfonos dentro del prompt o del agente.

Diseño:
- Para MVP se usa JSON local.
- La recuperación se hace por coincidencia simple de ciudad.
- En el futuro puede evolucionar a búsqueda geográfica, coordenadas,
  base de datos o CRM.
"""

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List


DEALERS_PATH = Path("data/dealers")


def load_dealers(brand: str, market: str) -> Dict[str, Any]:
    """
    Carga la base de concesionarios según marca y mercado.

    Args:
        brand (str): marca activa. Ej: "volvo".
        market (str): mercado activo. Ej: "colombia".

    Returns:
        Dict[str, Any]: base de dealers/concesionarios.
    """
    file_name = f"{brand.lower()}_{market.lower()}.json"
    file_path = DEALERS_PATH / file_name

    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_dealers_list(
    brand: str,
    market: str,
) -> List[Dict[str, Any]]:
    """
    Devuelve la lista plana de concesionarios para uso interno.

    Responsabilidad:
    - Reutilizar la carga existente de dealers.
    - Entregar datos estructurados para RAG V3.
    - Evitar duplicar loaders en otros servicios.

    Args:
        brand (str): marca activa. Ej: "volvo".
        market (str): mercado activo. Ej: "colombia".

    Returns:
        List[Dict[str, Any]]: lista de concesionarios.
    """
    dealer_data = load_dealers(brand=brand, market=market)

    if not dealer_data:
        return []

    return dealer_data.get("dealers", [])




def retrieve_dealer_context(
    user_message: str,
    lead_state: Dict[str, Any],
    brand: str,
    market: str,
    top_k: int = 3,
) -> str:
    """
    Recupera contexto relevante de concesionarios según ciudad.

    Responsabilidad:
    - Identificar la ciudad del cliente desde el estado del lead o mensaje.
    - Recuperar las sedes disponibles para esa ciudad.
    - Devolver contexto compacto para que el LLM oriente la visita o prueba.

    Args:
        user_message (str): mensaje actual del usuario.
        lead_state (Dict[str, Any]): estado actual del lead.
        brand (str): marca activa.
        market (str): mercado activo.
        top_k (int): máximo de sedes a incluir.

    Returns:
        str: contexto de concesionarios listo para agregar al prompt.
    """
    dealer_data = load_dealers(brand=brand, market=market)

    if not dealer_data:
        return ""

    city = str(lead_state.get("city", "") or "").lower()
    query = f"{user_message} {city}".lower()

    scored_dealers = []

    for dealer in dealer_data.get("dealers", []):
        score = 0

        dealer_city = str(dealer.get("city", "")).lower()
        dealer_name = str(dealer.get("name", "")).lower()

        # -----------------------------
        # Match principal por ciudad
        # -----------------------------
        if dealer_city and dealer_city in query:
            score += 5

        # -----------------------------
        # Match secundario por nombre de sede
        # -----------------------------
        if dealer_name and dealer_name in query:
            score += 3

        # -----------------------------
        # Señales de intención comercial
        # -----------------------------
        if any(word in query for word in ["prueba", "test drive", "visita", "vitrina", "agendar"]):
            if "sales" in dealer.get("type", []):
                score += 2

        if any(word in query for word in ["taller", "servicio", "postventa", "mantenimiento"]):
            if "service" in dealer.get("type", []):
                score += 2

        if score > 0:
            scored_dealers.append((score, dealer))

    scored_dealers.sort(key=lambda item: item[0], reverse=True)

    selected_dealers = [dealer for _, dealer in scored_dealers[:top_k]]

    if not selected_dealers:
        return ""

    lines = ["CONCESIONARIOS / SEDES RELEVANTES:"]

    for dealer in selected_dealers:
        name = dealer.get("name", "")
        city = dealer.get("city", "")
        phone = dealer.get("phone", "")
        dealer_type = ", ".join(dealer.get("type", []))
        notes = dealer.get("notes", "")

        address = dealer.get("address", "")
        sales_address = dealer.get("sales_address", "")
        service_address = dealer.get("service_address", "")

        lines.append(f"\n- Sede: {name}")
        lines.append(f"  Ciudad: {city}")
        lines.append(f"  Tipo: {dealer_type}")

        if address:
            lines.append(f"  Dirección: {address}")

        if sales_address:
            lines.append(f"  Dirección ventas: {sales_address}")

        if service_address:
            lines.append(f"  Dirección taller: {service_address}")

        if phone:
            lines.append(f"  Teléfono: {phone}")

        if notes:
            lines.append(f"  Nota: {notes}")

    return "\n".join(lines)

def normalize_text(value: str) -> str:
    """
    Normaliza texto para comparar ciudades sin depender de mayúsculas,
    tildes o espacios adicionales.
    """
    value = value or ""
    value = value.lower().strip()

    value = unicodedata.normalize("NFD", value)
    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Mn"
    )

    return value


def detect_city_from_message(
    user_message: str,
    dealers: List[Dict[str, Any]],
) -> str:
    """
    Detecta si el mensaje actual menciona una ciudad que existe
    en el catálogo de dealers.

    Esto permite casos como:
    - cliente vive en Popayán
    - pero dice: "Cali me queda bien"

    En ese caso, el contexto de sedes debe priorizar Cali para la visita.
    """
    message_norm = normalize_text(user_message)

    for dealer in dealers:
        dealer_city = dealer.get("city", "")
        dealer_city_norm = normalize_text(dealer_city)

        if dealer_city_norm and dealer_city_norm in message_norm:
            return dealer_city

    return ""


def build_dealer_context_for_prompt(
    dealers: List[Dict[str, Any]],
    lead_state: Dict[str, Any],
    user_message: str = "",
) -> str:
    """
    Construye contexto compacto de concesionarios para el prompt.

    Responsabilidad:
    - Entregar al LLM información real de sedes disponibles.
    - Priorizar la ciudad mencionada en el mensaje actual si existe.
    - Si no hay ciudad mencionada, usar la ciudad de la cita.
    - Si no hay ciudad de cita, usar la ciudad del cliente.
    - Evitar enviar todas las direcciones completas cuando no son necesarias.

    Importante:
    Esta función NO calcula cercanía geográfica.
    Si no hay sede exacta para la ciudad relevante, no debe inventarse sede local.
    """

    if not dealers:
        return "No hay concesionarios cargados en el catálogo."

    # -----------------------------
    # 1. Detectar ciudad relevante
    # -----------------------------
    message_city = detect_city_from_message(
        user_message=user_message,
        dealers=dealers,
    )

    appointment_location = lead_state.get("appointment_location") or {}
    appointment_city = ""

    if isinstance(appointment_location, dict):
        appointment_city = appointment_location.get("city", "") or ""

    lead_city = lead_state.get("city", "") or ""

    # Prioridad:
    # 1. ciudad mencionada en el mensaje actual
    # 2. ciudad de la cita
    # 3. ciudad del cliente
    context_city = message_city or appointment_city or lead_city
    context_city_norm = normalize_text(context_city)

    # -----------------------------
    # 2. Buscar sedes exactas
    # -----------------------------
    exact_city_dealers = [
        dealer
        for dealer in dealers
        if normalize_text(dealer.get("city", "")) == context_city_norm
    ]

    # -----------------------------
    # 3. Si hay ciudad exacta, enviar solo esas sedes
    # -----------------------------
    if context_city and exact_city_dealers:
        lines = [
            f"SEDES REGISTRADAS PARA {context_city}:",
            "Usa solo estas sedes para direcciones o visitas en esta ciudad.",
        ]

        if len(exact_city_dealers) > 1:
            lines.append(
                "Hay varias sedes disponibles en esta ciudad. Antes de confirmar una cita, "
                "el cliente debe elegir una sede o el asesor debe proponer una sede específica."
            )
        for dealer in exact_city_dealers:
            dealer_type = ", ".join(dealer.get("type", []))

            lines.append(f"\n- {dealer.get('name', '')}")
            lines.append(f"  Ciudad: {dealer.get('city', '')}")
            lines.append(f"  Tipo: {dealer_type}")

            if dealer.get("address"):
                lines.append(f"  Dirección: {dealer.get('address')}")

            if dealer.get("sales_address"):
                lines.append(f"  Dirección ventas: {dealer.get('sales_address')}")

            if dealer.get("service_address"):
                lines.append(f"  Dirección taller: {dealer.get('service_address')}")

            if dealer.get("phone"):
                lines.append(f"  Teléfono: {dealer.get('phone')}")

        return "\n".join(lines)
# -----------------------------
    # 4. Si no hay sede exacta, enviar cobertura compacta
    # -----------------------------
    available_cities = sorted(
        {
            dealer.get("city", "")
            for dealer in dealers
            if dealer.get("city")
        }
    )

    lines = []

    if context_city:
        lines.append(
            f"No hay sede registrada exactamente para {context_city}."
        )
    else:
        lines.append(
            "No hay ciudad de visita confirmada."
        )

    lines.append(
        "Ciudades con sedes registradas en el catálogo:"
    )
    lines.append(", ".join(available_cities))

    lines.append(
        "Nota crítica: no inventar sede, dirección, prueba de manejo ni entrega local "
        "en una ciudad que no aparezca como sede registrada."
    )

    lines.append(
        "Próxima acción permitida: preguntar al cliente qué ciudad con sede registrada "
        "le queda mejor para la visita, o proponer revisar opciones disponibles."
    )

    lines.append(
        "No confirmar ni insinuar visita local en la ciudad del cliente si no hay sede exacta."
    )

    return "\n".join(lines)