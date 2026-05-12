"""
Módulo: ollama_service

Responsabilidad:
Encapsular la comunicación con el motor de IA local (Ollama).

Este servicio permite:
- enviar prompts al modelo LLM,
- recibir la respuesta generada,
- manejar errores de conexión o timeout,
- evitar que el backend falle si el modelo no responde.

Beneficio:
El resto del sistema (agentes, API) no depende de detalles de implementación
de Ollama. Solo consume esta función.

Diseño:
- Centraliza configuración de modelo
- Permite cambiar modelo fácilmente (testing vs producción)
- Implementa fallback seguro en caso de error
"""

import requests
import time
from app.utils.debug import debug_timer_start, debug_timer_end


# =========================
# CONFIGURACIÓN
# =========================

# Endpoint local de Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# Modelo activo
# Para pruebas usar:
# - "phi" (rápido)
# - "mistral" (balance)
# Para producción:
# - "qwen2.5:3b"
OLLAMA_MODEL = "gemma3:4b"


# =========================
# FUNCIÓN PRINCIPAL
# =========================

def generate_with_ollama(prompt: str) -> str:
    """
    Envía un prompt al modelo de Ollama y retorna el texto generado.

    Flujo:
    1. construye el payload requerido por Ollama,
    2. realiza llamada HTTP POST,
    3. valida respuesta,
    4. retorna el texto generado,
    5. si falla, devuelve respuesta segura (fallback JSON).

    Args:
        prompt (str):
            Instrucción completa enviada al modelo.

    Returns:
        str:
            Texto generado por el modelo.
            IMPORTANTE: siempre retorna un string válido,
            incluso en caso de error (fallback controlado).

    Notas:
        - Se usa stream=False para obtener respuesta completa.
        - Se reduce timeout para evitar bloqueos largos.
        - Se evita lanzar excepción para no romper FastAPI.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        # =========================
        # Llamada HTTP
        # =========================

        start = debug_timer_start()

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        debug_timer_end(start, "Ollama request")

        response.raise_for_status()

        # Asegurar correcta codificación
        response.encoding = "utf-8"

        data = response.json()

        # =========================
        # Validación de respuesta
        # =========================
        if "response" not in data:
            raise ValueError("Respuesta sin campo 'response'")

        return data["response"]

    except Exception as e:
        # =========================
        # FALLBACK CONTROLADO
        # =========================
        # ⚠️ NO lanzar excepción → evita romper toda la app
        print(f"⚠️ Error Ollama: {e}")

        return """
{
  "assistant_reply": "Estoy teniendo un inconveniente técnico temporal con el motor de IA. Intentemos nuevamente en unos segundos.",
  "updated_lead_state": {},
  "next_action": "continue_conversation",
  "confidence": 0.0
}
"""