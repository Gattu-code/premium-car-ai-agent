# -----------------------------
# Vector Retrieval Service
# -----------------------------
# Responsabilidad:
# Recuperar modelos relevantes usando embeddings locales.
#
# Esto reemplaza el matching por palabras clave por búsqueda semántica.
#
# Principio:
# - El backend NO recomienda.
# - El backend SOLO recupera contexto relevante.
# - El LLM razona con ese contexto.

import numpy as np
from typing import Dict, List

from app.services.embedding_service import generate_embedding


# -----------------------------
# Cache simple en memoria
# -----------------------------
# Para V2 mantenemos esto simple.
# Más adelante puede migrarse a Chroma, FAISS o base de datos vectorial.
_VECTOR_INDEX: list[dict] = []


def build_model_text(model: Dict) -> str:
    """
    Convierte un modelo del portafolio en texto semántico rico.

    Este texto es el que se vectoriza.

    Args:
        model (Dict): modelo del portafolio.

    Returns:
        str: descripción optimizada para embeddings.
    """
    versions = model.get("versions", [])

    version_text = ", ".join(
        [
            version.get("name", "")
            for version in versions
            if version.get("name")
        ]
    )

    best_for = ", ".join(model.get("best_for", []))
    strengths = ", ".join(model.get("key_strengths", []))

    return f"""
    Marca: {model.get("brand", "")}
    Mercado: {model.get("market", "")}

    Modelo: {model.get("model", "")}
    Tipo de carrocería: {model.get("body_type", "")}
    Segmento: {model.get("segment", "")}
    Tecnología / combustible: {model.get("fuel_type", "")}

    Es completamente eléctrico: {model.get("is_fully_electric", "")}
    Precio desde COP: {model.get("price_from_cop", "")}
    Autonomía aproximada km: {model.get("range_km", "")}
    Tiempo de carga: {model.get("charging_time", "")}

    Versiones disponibles: {version_text}

    Uso ideal: {best_for}
    Fortalezas principales: {strengths}

    Enfoque familiar: {model.get("family_fit", "")}
    Uso urbano: {model.get("urban_fit", "")}
    Uso para viajes: {model.get("travel_fit", "")}
    Nivel premium: {model.get("premium_level", "")}

    Descripción comercial:
    {model.get("description", "")}

    Perfil de cliente recomendado:
    Este modelo puede ser relevante para clientes que buscan {best_for},
    valoran {strengths}, y necesitan un vehículo del segmento {model.get("segment", "")}.
    """.strip()


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calcula similitud coseno entre dos vectores.

    Args:
        vector_a (list[float]): vector consulta.
        vector_b (list[float]): vector documento.

    Returns:
        float: similitud entre 0 y 1 aproximadamente.
    """
    if not vector_a or not vector_b:
        return 0.0

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def build_query_text(user_message: str, lead_state: Dict) -> str:
    """
    Construye texto de búsqueda combinando mensaje actual y estado del lead.

    Args:
        user_message (str): mensaje actual del cliente.
        lead_state (Dict): información acumulada del lead.

    Returns:
        str: consulta semántica.
    """
    return f"""
    Mensaje actual del cliente:
    {user_message}

    Datos acumulados del lead:
    Nombre: {lead_state.get("lead_name", "")}
    Ciudad: {lead_state.get("city", "")}
    Interés: {lead_state.get("vehicle_interest", "")}
    Segmento: {lead_state.get("vehicle_segment", "")}
    Presupuesto: {lead_state.get("budget_range", "")}
    Forma de pago: {lead_state.get("payment_method", "")}
    Tiempo de compra: {lead_state.get("purchase_timeframe", "")}
    Contexto familiar: {lead_state.get("family_context", "")}
    Motivación principal: {lead_state.get("primary_motivation", "")}
    """.strip()


def build_vector_index(catalog: List[Dict]) -> None:
    """
    Construye índice vectorial en memoria para el portafolio.

    Args:
        catalog (List[Dict]): portafolio activo.
    """
    global _VECTOR_INDEX

    _VECTOR_INDEX = []

    for model in catalog:
        model_text = build_model_text(model)
        embedding = generate_embedding(model_text)

        _VECTOR_INDEX.append(
            {
                "type": "model",
                "model": model,
                "text": model_text,
                "embedding": embedding
            }
        )


def retrieve_relevant_models_vector(
    user_message: str,
    lead_state: Dict,
    catalog: List[Dict],
    top_k: int = 3
) -> List[Dict]:
    """
    Recupera modelos relevantes usando embeddings.

    Args:
        user_message (str): mensaje actual del cliente.
        lead_state (Dict): estado acumulado del lead.
        catalog (List[Dict]): portafolio activo.
        top_k (int): cantidad de modelos a recuperar.

    Returns:
        List[Dict]: modelos más relevantes.
    """
    global _VECTOR_INDEX

    if not _VECTOR_INDEX:
        build_vector_index(catalog)

    query_text = build_query_text(user_message, lead_state)
    query_embedding = generate_embedding(query_text)

    scored_items = []

    for item in _VECTOR_INDEX:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_items.append((score, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)

    return [
        item["model"]
        for score, item in scored_items[:top_k]
        if score > 0
    ]