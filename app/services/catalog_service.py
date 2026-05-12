"""
Módulo: catalog_service

Responsabilidad:
Cargar el portafolio activo de vehículos y recuperar modelos relevantes
para construir contexto que será entregado al LLM.

Objetivo:
- Mantener el conocimiento de producto en archivos de datos.
- Evitar reglas rígidas de recomendación dentro del backend.
- Preparar la arquitectura para multi-marca, multi-mercado y RAG semántico futuro.

Diseño:
- Para MVP se usa JSON local.
- La recuperación se hace con búsqueda simple por texto y señales suaves.
- En el futuro puede evolucionar a embeddings + vector database.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


CATALOGS_PATH = Path("data/catalogs")


def load_catalog(brand: str, market: str) -> List[Dict]:
    """
    Carga el portafolio activo según marca y mercado.

    Args:
        brand (str): marca del portafolio. Ej: "volvo".
        market (str): mercado del portafolio. Ej: "colombia".

    Returns:
        List[Dict]: lista de modelos disponibles en el portafolio.
    """
    file_name = f"{brand.lower()}_{market.lower()}.json"
    file_path = CATALOGS_PATH / file_name

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_search_text(model: Dict) -> str:
    """
    Convierte un modelo del portafolio en texto buscable.

    Responsabilidad:
    - Transformar atributos estructurados del JSON en una representación textual.
    - Facilitar retrieval simple sin depender todavía de embeddings.
    - Mantener el conocimiento en datos, no en reglas quemadas en código.

    Args:
        model (Dict): modelo individual del portafolio.

    Returns:
        str: texto normalizado que representa el modelo.
    """
    fields = [
        model.get("model", ""),
        model.get("body_type", ""),
        model.get("segment", ""),
        model.get("fuel_type", ""),
        model.get("description", ""),
        model.get("family_fit", ""),
        model.get("urban_fit", ""),
        model.get("travel_fit", ""),
        model.get("premium_level", ""),
        " ".join(model.get("best_for", [])),
        " ".join(model.get("key_strengths", [])),
    ]

    versions = model.get("versions", [])

    for version in versions:
        fields.extend(
            [
                version.get("name", ""),
                version.get("range_level", ""),
                version.get("power_level", ""),
                version.get("focus", ""),
            ]
        )

    return " ".join(str(field) for field in fields if field).lower()


def build_query_text(user_message: str, lead_state: Dict) -> str:
    """
    Construye el texto de búsqueda para recuperar modelos relevantes.

    Combina:
    - mensaje actual del usuario,
    - estado acumulado del lead.

    Args:
        user_message (str): mensaje actual del cliente.
        lead_state (Dict): estado actual del lead.

    Returns:
        str: texto consolidado para búsqueda.
    """
    lead_parts = [
        lead_state.get("vehicle_interest", ""),
        lead_state.get("vehicle_segment", ""),
        lead_state.get("budget_range", ""),
        lead_state.get("lifestyle_profile", ""),
        lead_state.get("family_context", ""),
        lead_state.get("primary_motivation", ""),
        lead_state.get("interest_type", ""),
        lead_state.get("purchase_timeframe", ""),
        lead_state.get("current_vehicle", ""),
    ]

    return " ".join(
        [user_message] + [str(part) for part in lead_parts if part]
    ).lower()


def retrieve_relevant_models(
    user_message: str,
    lead_state: Dict,
    catalog: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """
    Recupera modelos relevantes del portafolio usando retrieval simple.

    Importante:
    - Esta función NO decide qué vehículo debe recomendarse.
    - Solo recupera contexto relevante para que el LLM razone.
    - Es una versión RAG-ready sin embeddings.

    Estrategia:
    1. Construye una consulta combinando mensaje + estado del lead.
    2. Detecta si el usuario pregunta por una categoría completa.
    3. Convierte cada modelo en texto buscable.
    4. Calcula coincidencias simples por tokens.
    5. Devuelve los modelos más relevantes.

    Args:
        user_message (str): mensaje actual del cliente.
        lead_state (Dict): estado actual del lead.
        catalog (List[Dict]): portafolio activo.
        top_k (int): número máximo de modelos a devolver.

    Returns:
        List[Dict]: modelos relevantes para construir contexto.
    """
    query_text = build_query_text(user_message, lead_state)
    query_tokens = set(query_text.split())

    # -----------------------------
    # Retrieval por intención de inventario
    # -----------------------------
    # Cuando el cliente pregunta por una categoría completa
    # (ej: híbridos, eléctricos), conviene devolver todos los modelos
    # que cumplen ese criterio, no solo los modelos con mejor score.
    #
    # Importante:
    # - Esto NO decide qué vehículo recomendar.
    # - Solo garantiza que el LLM vea el portafolio completo
    #   de la categoría solicitada.
    # - Es útil para preguntas como:
    #   "¿cuáles son híbridos?"
    #   "¿solo existe ese híbrido?"
    #   "¿qué eléctricos tienen?"
    if any(
        word in query_text
        for word in ["híbrido", "hibrido", "híbridos", "hibridos", "enchufable", "plug-in", "plugin"]
    ):
        hybrid_models = [
            model
            for model in catalog
            if (
                "híbrido" in model.get("fuel_type", "").lower()
                or "hibrido" in model.get("fuel_type", "").lower()
                or "enchufable" in model.get("fuel_type", "").lower()
            )
        ]

        if hybrid_models:
            return hybrid_models

    if any(
        word in query_text
        for word in ["eléctrico", "electrico", "eléctricos", "electricos"]
    ):
        electric_models = [
            model
            for model in catalog
            if model.get("is_fully_electric") is True
        ]

        if electric_models:
            return electric_models

    # -----------------------------
    # Scoring de modelos
    # -----------------------------
    scored_models = []

    for model in catalog:
        model_text = build_model_search_text(model)
        score = 0

        # -----------------------------
        # Retrieval por coincidencia simple
        # -----------------------------
        for token in query_tokens:
            if len(token) > 2 and token in model_text:
                score += 1

        # -----------------------------
        # Señales suaves desde el lead
        # -----------------------------
        # Estas señales no deciden el vehículo.
        # Solo ayudan a priorizar contexto relevante.
        if "familia" in query_text and model.get("family_fit") in ["alta", "muy alta"]:
            score += 2

        if any(word in query_text for word in ["ciudad", "urbano", "diario"]):
            if model.get("urban_fit") in ["alta", "muy alta"]:
                score += 2

        if any(word in query_text for word in ["viaje", "viajes", "carretera"]):
            if model.get("travel_fit") in ["alta", "muy alta"]:
                score += 2

        if any(word in query_text for word in ["eléctrico", "electrico"]):
            if model.get("is_fully_electric") is True:
                score += 2

        if any(word in query_text for word in ["híbrido", "hibrido"]):
            if "híbrido" in model.get("fuel_type", "").lower():
                score += 2

        scored_models.append((score, model))

    # -----------------------------
    # Ordenar por relevancia
    # -----------------------------
    scored_models.sort(key=lambda item: item[0], reverse=True)

    # -----------------------------
    # Seleccionar modelos con score positivo
    # -----------------------------
    relevant_models = [
        model for score, model in scored_models if score > 0
    ][:top_k]

    # -----------------------------
    # Fallback
    # -----------------------------
    # Si no hay coincidencias, enviamos pocos modelos para que el LLM
    # pueda orientar sin saturar el prompt.
    if not relevant_models:
        return catalog[:top_k]

    return relevant_models


def build_catalog_context(catalog: List[Dict[str, Any]]) -> str:
    """
    Construye contexto compacto del catálogo activo.

    Responsabilidad:
    - Entregar al LLM todos los modelos disponibles.
    - Mantener datos críticos como tipo, segmento, energía, precio y autonomía.
    - Evitar enviar JSON largo o campos innecesarios.
    """

    if not catalog:
        return "No hay modelos cargados en el catálogo."

    lines = [
        "CATÁLOGO ACTIVO COMPLETO:",
        "Usa únicamente estos modelos. No inventes modelos fuera de esta lista.",
    ]

    for model in catalog:
        model_name = model.get("model", "")
        segment = model.get("segment", "")
        body_type = model.get("body_type", "")
        fuel_type = model.get("fuel_type", "")
        price = model.get("price_from_cop")
        range_km = model.get("range_km") or model.get("range_electric_km")
        seating = model.get("seating_capacity")
        best_for = ", ".join(model.get("best_for", [])[:5])
        strengths = ", ".join(model.get("key_strengths", [])[:5])
        description = model.get("description", "")

        lines.append(f"\nModelo: {model_name}")
        lines.append(f"- Tipo: {body_type}")
        lines.append(f"- Segmento: {segment}")
        lines.append(f"- Energía: {fuel_type}")

        if price:
            lines.append(f"- Precio desde: {price:,} COP".replace(",", "."))

        if range_km:
            lines.append(f"- Autonomía: {range_km} km")

        if seating:
            lines.append(f"- Plazas: {seating}")

        if best_for:
            lines.append(f"- Ideal para: {best_for}")

        if strengths:
            lines.append(f"- Fortalezas: {strengths}")

        if description:
            lines.append(f"- Descripción: {description}")

    return "\n".join(lines)