"""
Módulo: ai_provider

Responsabilidad:
Ser el punto único de entrada para generar respuestas con IA.

Este módulo permite cambiar el motor de IA sin modificar el agente:
- Ollama local
- API externa genérica
"""

from app.config import AI_PROVIDER
from app.services.ollama_service import generate_with_ollama
from app.services.external_llm_service import generate_with_external_llm


def generate_ai_response(prompt: str) -> str:
    """
    Genera una respuesta usando el proveedor configurado.

    Args:
        prompt (str): prompt final construido por el agente.

    Returns:
        str: respuesta cruda del proveedor, idealmente JSON válido.
    """

    if AI_PROVIDER == "ollama":
        return generate_with_ollama(prompt)

    if AI_PROVIDER == "external":
        return generate_with_external_llm(prompt)

    raise ValueError(f"Proveedor IA no soportado: {AI_PROVIDER}")