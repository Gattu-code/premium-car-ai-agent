# -----------------------------
# Vector Retrieval Service V3
# -----------------------------
# Responsabilidad:
# Implementar recuperación semántica multi-fuente usando embeddings locales.
#
# Este servicio construye un índice vectorial en memoria a partir de varias
# fuentes controladas del sistema:
# - portafolio de modelos
# - sedes / concesionarios
# - conocimiento de marca
# - reglas internas de negocio
#
# Objetivo:
# - Recuperar el contexto más relevante antes de llamar al LLM.
# - Reducir alucinaciones entregando información controlada.
# - Permitir que el agente responda con mayor precisión sobre modelos,
#   sedes, seguridad, financiación y reglas comerciales.
#
# Principio de diseño:
# - Este servicio NO recomienda vehículos.
# - Este servicio NO genera respuestas.
# - Este servicio NO modifica el estado del lead.
# - Este servicio SOLO recupera contexto relevante.
#
# Diferencia vs V2:
# - V2 recupera únicamente modelos del portafolio.
# - V3 recupera contexto desde múltiples fuentes.
#
# Arquitectura:
# user_message + lead_state
# → query semántica
# → embedding local
# → comparación contra índice vectorial
# → contexto agrupado por tipo
# → prompt final del LLM

from typing import Any, Dict, List
import numpy as np

from app.services.embedding_service import generate_embedding


_VECTOR_INDEX_V3: List[Dict[str, Any]] = []


import unicodedata


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación semántica básica.

    Responsabilidad:
    - Convertir texto a minúsculas.
    - Eliminar acentos y caracteres diacríticos.
    - Preparar strings para comparaciones consistentes.

    Importante:
    Esta función NO debe usarse para modificar:
    - mensajes del usuario,
    - prompts,
    - respuestas del modelo.

    Solo se usa para:
    - filtros de negocio,
    - matching de atributos (ej: ciudad).

    Args:
        text (str): texto original.

    Returns:
        str: texto normalizado.
    """
    if not text:
        return ""

    text = text.lower().strip()

    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

def detect_explicit_models(
    user_message: str,
    catalog: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detecta modelos mencionados explícitamente por el usuario.

    Responsabilidad:
    - Identificar referencias exactas a modelos del catálogo.
    - Evitar que embeddings excluya modelos mencionados por nombre.
    - Mantener la lógica genérica, sin reglas específicas por modelo.

    Importante:
    Esta función NO decide qué recomendar.
    Solo garantiza que un modelo mencionado explícitamente entre al contexto.

    Args:
        user_message (str): mensaje actual del cliente.
        catalog (List[Dict[str, Any]]): catálogo activo.

    Returns:
        List[Dict[str, Any]]: modelos detectados explícitamente.
    """
    message = normalize_text(user_message)
    detected_models = []

    for model in catalog:
        model_name = str(model.get("model", ""))
        model_key = normalize_text(model_name)

        if model_key and model_key in message:
            detected_models.append(model)

    return detected_models



# -----------------------------
# Utilidades
# -----------------------------
def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    """
    Calcula similitud coseno entre dos embeddings.

    Args:
        vector_a (List[float]): embedding de consulta.
        vector_b (List[float]): embedding del documento.

    Returns:
        float: score de similitud.
    """
    if not vector_a or not vector_b:
        return 0.0

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# -----------------------------
# Builders de texto para embeddings
# -----------------------------
def build_model_text(model: Dict[str, Any]) -> str:
    """
    Construye una representación textual enriquecida de un modelo del portafolio.

    Esta función convierte un objeto estructurado del catálogo en texto natural
    para que pueda ser convertido en embedding. El objetivo no es mostrar este
    texto al usuario, sino mejorar la recuperación semántica.

    Un buen texto de modelo debe permitir que el embedding entienda:
    - qué tipo de vehículo es,
    - para qué perfil de cliente aplica,
    - qué tecnología usa,
    - qué fortalezas comerciales tiene,
    - y en qué casos debería ser considerado relevante.

    Args:
        model (Dict[str, Any]): modelo del portafolio cargado desde el JSON.

    Returns:
        str: texto semántico rico optimizado para embeddings.
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
    Fuente: portafolio de modelos.
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
    Potencia HP: {model.get("power_hp", "")}
    Capacidad de pasajeros: {model.get("seating_capacity", "")}

    Versiones disponibles: {version_text}

    Uso ideal: {best_for}
    Fortalezas principales: {strengths}

    Enfoque familiar: {model.get("family_fit", "")}
    Uso urbano: {model.get("urban_fit", "")}
    Uso para viajes: {model.get("travel_fit", "")}
    Nivel premium: {model.get("premium_level", "")}

    Descripción comercial:
    {model.get("description", "")}
    """.strip()


def build_dealer_text(dealer: Dict[str, Any]) -> str:
    """
    Construye una representación textual de una sede o concesionario.

    Esta función permite que el RAG recupere sedes relevantes cuando el usuario
    menciona ciudad, intención de visita, prueba de manejo, taller o postventa.

    El texto generado ayuda al embedding a relacionar:
    - ciudad,
    - nombre de sede,
    - tipo de atención,
    - dirección,
    - teléfono,
    - y casos de uso comercial.

    Args:
        dealer (Dict[str, Any]): sede/concesionario cargado desde JSON.

    Returns:
        str: texto semántico de la sede para indexación vectorial.
    """
    
    dealer_type = ", ".join(dealer.get("type", []))

    return f"""
    Fuente: concesionarios y sedes.
    Nombre de sede: {dealer.get("name", "")}
    Ciudad: {dealer.get("city", "")}
    Tipo de atención: {dealer_type}

    Dirección general: {dealer.get("address", "")}
    Dirección ventas: {dealer.get("sales_address", "")}
    Dirección taller: {dealer.get("service_address", "")}
    Teléfono: {dealer.get("phone", "")}

    Notas:
    {dealer.get("notes", "")}

    Esta sede puede ser relevante para clientes que quieran visitar vitrina,
    agendar prueba de manejo, recibir asesoría comercial o consultar postventa
    en la ciudad indicada.
    """.strip()


def build_brand_knowledge_text(item: Dict[str, Any]) -> str:
    """
    Construye una representación textual de un tema de conocimiento de marca.

    Esta función transforma temas como seguridad, electrificación, baterías,
    tecnología o historia de marca en texto indexable por embeddings.

    Sirve para que el agente pueda responder preguntas que no dependen
    directamente de un modelo específico, pero sí del conocimiento controlado
    de la marca.

    Args:
        item (Dict[str, Any]): tema de conocimiento de marca.

    Returns:
        str: texto semántico del tema para recuperación RAG.
    """
    
    keywords = ", ".join(item.get("keywords", []))

    return f"""
    Fuente: conocimiento de marca.
    Tema: {item.get("topic", "")}
    Palabras clave: {keywords}

    Información:
    {item.get("content", "")}
    """.strip()


def build_business_rule_text(rule: Dict[str, Any]) -> str:
    """
    Construye una representación textual de una regla interna de negocio.

    Esta función permite que reglas comerciales como teléfono obligatorio,
    no repetir datos, financiación o criterios de cierre estén disponibles
    para el RAG.

    Importante:
    Estas reglas no reemplazan la validación dura del backend. Sirven para
    reforzar el comportamiento del LLM dentro del prompt.

    Args:
        rule (Dict[str, Any]): regla interna estructurada.

    Returns:
        str: texto de regla optimizado para embeddings.
    """
    
    return f"""
    Fuente: reglas internas de negocio.
    Regla: {rule.get("name", "")}

    Descripción:
    {rule.get("description", "")}
    """.strip()


# -----------------------------
# Construcción del índice
# -----------------------------
def build_vector_index_v3(
    catalog: List[Dict[str, Any]],
    dealers: List[Dict[str, Any]],
    brand_knowledge: List[Dict[str, Any]],
    business_rules: List[Dict[str, Any]],
) -> None:
    """
    Construye el índice vectorial multi-fuente en memoria.

    Esta función toma todas las fuentes controladas, convierte cada elemento
    en texto semántico, genera su embedding y lo guarda en un índice en memoria.

    Para V3 usamos memoria local por simplicidad. En una versión futura este
    índice puede migrarse a Chroma, FAISS, PostgreSQL + pgvector o una base
    vectorial administrada.

    Args:
        catalog: modelos del portafolio.
        dealers: sedes/concesionarios.
        brand_knowledge: temas de conocimiento de marca.
        business_rules: reglas internas del negocio.
    """
    global _VECTOR_INDEX_V3

    _VECTOR_INDEX_V3 = []

    for model in catalog:
        text = build_model_text(model)

        _VECTOR_INDEX_V3.append(
            {
                "type": "model",
                "text": text,
                "embedding": generate_embedding(text),
                "data": model,
            }
        )

    for dealer in dealers:
        text = build_dealer_text(dealer)

        _VECTOR_INDEX_V3.append(
            {
                "type": "dealer",
                "text": text,
                "embedding": generate_embedding(text),
                "data": dealer,
            }
        )

    for item in brand_knowledge:
        text = build_brand_knowledge_text(item)

        _VECTOR_INDEX_V3.append(
            {
                "type": "brand_knowledge",
                "text": text,
                "embedding": generate_embedding(text),
                "data": item,
            }
        )

    for rule in business_rules:
        text = build_business_rule_text(rule)

        _VECTOR_INDEX_V3.append(
            {
                "type": "business_rule",
                "text": text,
                "embedding": generate_embedding(text),
                "data": rule,
            }
        )


# -----------------------------
# Query builder
# -----------------------------
def build_query_text(
    user_message: str,
    lead_state: Dict[str, Any],
) -> str:
    """
    Construye la consulta semántica usada para buscar en el índice vectorial.

    Combina el mensaje actual del usuario con el estado acumulado del lead.
    Esto es clave porque una misma frase puede significar cosas distintas
    según el contexto previo.

    Ejemplo:
    - Mensaje: "quiero probarlo"
    - Lead state: modelo XC60, ciudad Pereira
    → el RAG puede recuperar sede, regla de teléfono y contexto de agendamiento.

    Args:
        user_message: mensaje actual del cliente.
        lead_state: estado acumulado del lead.

    Returns:
        str: consulta enriquecida para generar embedding.
    """
    return f"""
    Mensaje actual del cliente:
    {user_message}

    Estado acumulado del lead:
    Nombre: {lead_state.get("lead_name", "")}
    Ciudad: {lead_state.get("city", "")}
    Marca: {lead_state.get("brand", "")}
    Interés de vehículo: {lead_state.get("vehicle_interest", "")}
    Segmento: {lead_state.get("vehicle_segment", "")}
    Presupuesto: {lead_state.get("budget_range", "")}
    Forma de pago: {lead_state.get("payment_method", "")}
    Plazo de compra: {lead_state.get("purchase_timeframe", "")}
    Contexto familiar: {lead_state.get("family_context", "")}
    Motivación principal: {lead_state.get("primary_motivation", "")}
    Intención: {lead_state.get("interest_type", "")}
    """.strip()


# -----------------------------
# Retrieval principal
# -----------------------------
def retrieve_relevant_context_v3(
    user_message: str,
    lead_state: Dict[str, Any],
    catalog: List[Dict[str, Any]],
    dealers: List[Dict[str, Any]],
    brand_knowledge: List[Dict[str, Any]],
    business_rules: List[Dict[str, Any]],
    top_k_models: int = 5,
    top_k_dealers: int = 2,
    top_k_brand: int = 2,
    top_k_rules: int = 3,
) -> Dict[str, Any]:
    """
    Recupera contexto relevante multi-fuente usando embeddings.
    
    Esta es la función principal del RAG V3. Recibe el mensaje actual,
    el estado del lead y las fuentes disponibles. Luego:
    1. construye o reutiliza el índice vectorial,
    2. genera embedding de la consulta,
    3. calcula similitud contra cada documento indexado,
    4. agrupa los resultados por tipo,
    5. devuelve contexto listo para el prompt.
    
    Args:
        user_message (str): mensaje actual del cliente.
        lead_state (Dict[str, Any]): estado acumulado del lead.
        catalog (List[Dict[str, Any]]): modelos del portafolio.
        dealers (List[Dict[str, Any]]): concesionarios/sedes.
        brand_knowledge (List[Dict[str, Any]]): conocimiento de marca.
        business_rules (List[Dict[str, Any]]): reglas internas.
        top_k_models (int): cantidad máxima de modelos.
        top_k_dealers (int): cantidad máxima de sedes.
        top_k_brand (int): cantidad máxima de temas de marca.
        top_k_rules (int): cantidad máxima de reglas.

    Returns:
        
        - models: modelos relevantes.
        - dealers: sedes relevantes.
        - brand_context: conocimiento de marca relevante.
        - rules_context: reglas internas relevantes.
        Dict[str, Any]: contexto recuperado agrupado por tipo.
    """
    global _VECTOR_INDEX_V3

    if not _VECTOR_INDEX_V3:
        build_vector_index_v3(
            catalog=catalog,
            dealers=dealers,
            brand_knowledge=brand_knowledge,
            business_rules=business_rules,
        )

    # -----------------------------
    # Detección de modelos explícitos
    # -----------------------------
    explicit_models = detect_explicit_models(
        user_message=user_message,
        catalog=catalog,
    )

    # -----------------------------
    # Prioridad a modelo del lead_state
    # -----------------------------
    lead_model = lead_state.get("vehicle_interest") or ""
    lead_model_norm = normalize_text(lead_model)

    if lead_model_norm:
        for model in catalog:
            model_name = model.get("model", "")
            model_norm = normalize_text(model_name)

            if model_norm == lead_model_norm:
                if model not in explicit_models:
                    explicit_models.append(model)



    query_text = build_query_text(
        user_message=user_message,
        lead_state=lead_state,
    )

    query_embedding = generate_embedding(query_text)

    scored_by_type = {
        "model": [],
        "dealer": [],
        "brand_knowledge": [],
        "business_rule": [],
    }
    
    lead_city = normalize_text(lead_state.get("city", ""))
    for item in _VECTOR_INDEX_V3:
        item_type = item["type"]

        # -----------------------------
        # Filtro duro para dealers
        # -----------------------------
        # Si ya conocemos la ciudad del lead, solo permitimos
        # dealers de esa ciudad antes de calcular similitud semántica.
        # Esto evita que el RAG recupere sedes de otras ciudades
        # por similitud general de intención comercial.
        if item_type == "dealer" and lead_city:
            dealer_city = normalize_text(item["data"].get("city", ""))

            if lead_city not in dealer_city:
                continue  # 🔥 excluimos

        score = cosine_similarity(query_embedding, item["embedding"])
        scored_by_type[item_type].append((score, item))




    for item_type in scored_by_type:
        scored_by_type[item_type].sort(key=lambda x: x[0], reverse=True)

    
    semantic_models = [
        item["data"]
        for score, item in scored_by_type["model"][:top_k_models]
        if score > 0
    ]
    
    
    # -----------------------------
    # Detección de ciudad explícita
    # -----------------------------
    explicit_city = None
    user_message_norm = normalize_text(user_message)

    for item in _VECTOR_INDEX_V3:
        if item["type"] == "dealer":
            city = item["data"].get("city", "")
            city_norm = normalize_text(city)

            if city_norm and city_norm in user_message_norm:
                explicit_city = city
                break
    

    # -----------------------------
    # Prioridad a modelos explícitos
    # -----------------------------
    # Si el usuario menciona un modelo por nombre, ese modelo debe entrar
    # al contexto aunque embeddings no lo ubique dentro del top_k.
    # Luego complementamos con modelos semánticamente similares.
    models = explicit_models + [
        model
        for model in semantic_models
        if model.get("model") not in {
            explicit_model.get("model")
            for explicit_model in explicit_models
        }
    ]

    # Limitar después del ranking
    models = models[:top_k_models]


    dealers_out = [
    item["data"]
    for score, item in scored_by_type["dealer"][:top_k_dealers]
    if score > 0
    ]

    # -----------------------------
    # Prioridad a ciudad explícita
    # -----------------------------
    if explicit_city:
        dealers_out = [
            item["data"]
            for item in _VECTOR_INDEX_V3
            if item["type"] == "dealer"
            and item["data"].get("city") == explicit_city
        ][:top_k_dealers]

    brand_context = "\n\n".join(
        [
            item["text"]
            for score, item in scored_by_type["brand_knowledge"][:top_k_brand]
            if score > 0
        ]
    )

    rules_context = "\n\n".join(
        [
            item["text"]
            for score, item in scored_by_type["business_rule"][:top_k_rules]
            if score > 0
        ]
    )

    return {
        "models": models,
        "dealers": dealers_out,
        "brand_context": brand_context,
        "rules_context": rules_context,
    }