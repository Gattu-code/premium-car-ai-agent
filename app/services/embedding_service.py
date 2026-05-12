# -----------------------------
# Embedding Service
# -----------------------------
# Responsabilidad:
# Convertir textos de portafolio, marca o sedes en vectores numéricos.
#
# Diseño:
# - Usa Ollama local para embeddings.
# - El LLM principal puede seguir siendo OpenRouter/Gemini.
# - Este servicio NO genera respuestas, solo vectores.

import requests


OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"


def generate_embedding(text: str) -> list[float]:
    """
    Genera embedding local usando Ollama.

    Args:
        text (str): texto a convertir en vector.

    Returns:
        list[float]: vector numérico del texto.
    """
    if not text or not text.strip():
        return []

    response = requests.post(
        OLLAMA_EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    return data.get("embedding", [])