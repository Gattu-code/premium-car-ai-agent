"""
Módulo: brand_knowledge_service

Responsabilidad:
Recuperar contexto de conocimiento de marca para enriquecer el prompt del agente.

Objetivo:
- Separar conocimiento de marca del catálogo de productos.
- Permitir que el agente responda sobre temas como seguridad, tecnología,
  historia, electrificación o diferenciadores de marca.
- Preparar la arquitectura para multi-marca y RAG semántico futuro.

Diseño:
- Para MVP se usa JSON local.
- La recuperación se hace por keywords simples.
- En el futuro puede evolucionar a embeddings + vector database.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


BRAND_KNOWLEDGE_PATH = Path("data/brand_knowledge")


def load_brand_knowledge(brand: str, market: str) -> Dict[str, Any]:
    """
    Carga la base de conocimiento de marca según marca y mercado.

    Args:
        brand (str): marca activa. Ej: "volvo".
        market (str): mercado activo. Ej: "colombia".

    Returns:
        Dict[str, Any]: base de conocimiento de marca.
    """
    file_name = f"{brand.lower()}_{market.lower()}.json"
    file_path = BRAND_KNOWLEDGE_PATH / file_name

    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_brand_context(
    user_message: str,
    lead_state: Dict[str, Any],
    brand: str,
    market: str,
    top_k: int = 2,
) -> str:
    """
    Recupera contexto relevante de conocimiento de marca.

    Responsabilidad:
    - Identificar si el mensaje del usuario toca temas de marca.
    - Recuperar los bloques más relevantes.
    - Devolver un contexto compacto para el LLM.

    Args:
        user_message (str): mensaje actual del usuario.
        lead_state (Dict[str, Any]): estado actual del lead.
        brand (str): marca activa.
        market (str): mercado activo.
        top_k (int): número máximo de temas a incluir.

    Returns:
        str: contexto de marca listo para agregar al prompt.
    """
    knowledge = load_brand_knowledge(brand=brand, market=market)

    if not knowledge:
        return ""

    query = " ".join(
        [
            user_message,
            str(lead_state.get("primary_motivation", "")),
            str(lead_state.get("family_context", "")),
            str(lead_state.get("vehicle_interest", "")),
        ]
    ).lower()

    scored_topics = []

    for topic in knowledge.get("topics", []):
        score = 0
        keywords = topic.get("keywords", [])

        for keyword in keywords:
            if keyword.lower() in query:
                score += 1

        if score > 0:
            scored_topics.append((score, topic))

    scored_topics.sort(key=lambda item: item[0], reverse=True)

    selected_topics = [topic for _, topic in scored_topics[:top_k]]

    if not selected_topics:
        return ""

    lines = ["CONOCIMIENTO DE MARCA RELEVANTE:"]

    for topic in selected_topics:
        lines.append(f"\nTema: {topic.get('topic', '')}")

        for item in topic.get("content", []):
            lines.append(f"- {item}")

    return "\n".join(lines)

def get_brand_knowledge_list(
    brand: str,
    market: str,
) -> List[Dict[str, Any]]:
    """
    Devuelve la lista estructurada de conocimiento de marca para RAG V3.

    Responsabilidad:
    - Reutilizar la carga existente de brand knowledge.
    - Entregar datos estructurados para embeddings.
    - Evitar duplicar loaders en otros servicios.

    Args:
        brand (str): marca activa. Ej: "volvo".
        market (str): mercado activo. Ej: "colombia".

    Returns:
        List[Dict[str, Any]]: lista de temas de conocimiento de marca.
    """
    knowledge = load_brand_knowledge(brand=brand, market=market)

    if not knowledge:
        return []

    result = []

    for topic in knowledge.get("topics", []):
        result.append(
            {
                "topic": topic.get("topic", ""),
                "keywords": topic.get("keywords", []),
                "content": " ".join(topic.get("content", [])),
            }
        )

    return result

def build_brand_context_for_prompt(
    brand_knowledge: Any,
) -> str:
    """
    Construye contexto completo de conocimiento de marca para el prompt.

    Responsabilidad:
    - Entregar al LLM conocimiento de marca estructurado y controlado.
    - Evitar depender de retrieval cuando la base de conocimiento es pequeña.
    - Mantener el conocimiento de marca separado del catálogo y de dealers.
    - Soportar tanto formato dict como formato list.

    Importante:
    Esta función NO decide qué responder.
    Esta función NO reemplaza RAG para bases grandes.
    Para MVP, entrega todo el conocimiento de marca disponible.
    En el futuro, si el archivo crece mucho, se puede volver a RAG semántico.
    """

    if not brand_knowledge:
        return "No hay conocimiento de marca cargado."

    # -----------------------------
    # Normalización de estructura
    # -----------------------------
    # Algunos loaders pueden devolver:
    # - dict: {"brand": "...", "market": "...", "topics": [...]}
    # - list: [{topic...}, {topic...}]
    #
    # Esta normalización evita errores si cambia el formato interno.
    if isinstance(brand_knowledge, dict):
        brand = brand_knowledge.get("brand", "")
        market = brand_knowledge.get("market", "")
        topics = brand_knowledge.get("topics", [])

    elif isinstance(brand_knowledge, list):
        brand = ""
        market = ""
        topics = brand_knowledge

    else:
        return "Formato de conocimiento de marca no reconocido."

    if not topics:
        return "No hay temas de conocimiento de marca disponibles."

    lines = [
        "CONOCIMIENTO DE MARCA DISPONIBLE:"
    ]

    if brand:
        lines.append(f"Marca: {brand}")

    if market:
        lines.append(f"Mercado: {market}")

    for topic in topics:
        if not isinstance(topic, dict):
            continue

        topic_name = topic.get("topic", "")
        keywords = topic.get("keywords", [])
        content = topic.get("content", [])

        if topic_name:
            lines.append(f"\nTema: {topic_name}")

        if keywords:
            lines.append(f"Claves: {', '.join(keywords[:5])}")

        for item in content[:3]:
            lines.append(f"- {item}")

    return "\n".join(lines)