"""
Módulo: external_llm_service

Responsabilidad:
Centralizar llamadas a APIs externas de LLM.

Objetivo:
Permitir cambiar entre proveedores externos modificando solo la configuración,
sin tocar la lógica del agente.

Proveedores soportados inicialmente:
- gemini
- openai
- mistral
"""
import time
import requests
from app.utils.debug import debug_llm_response


from app.config import (
    EXTERNAL_LLM_PROVIDER,
    EXTERNAL_LLM_API_KEY,
    EXTERNAL_LLM_MODEL,
)


def generate_with_external_llm(prompt: str) -> str:
    """
    Genera una respuesta usando el proveedor externo configurado.

    Args:
        prompt (str): prompt final construido por el agente.

    Returns:
        str: respuesta textual del proveedor.
    """

    if EXTERNAL_LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)

    if EXTERNAL_LLM_PROVIDER == "openai":
        return _call_openai(prompt)

    if EXTERNAL_LLM_PROVIDER == "mistral":
        return _call_mistral(prompt)
    
    if EXTERNAL_LLM_PROVIDER == "openrouter":
       return _call_openrouter(prompt)

    raise ValueError(f"Proveedor externo no soportado: {EXTERNAL_LLM_PROVIDER}")


def _fallback_response(error: Exception) -> str:
    """
    Devuelve una respuesta segura si falla el proveedor externo.

    Esto evita que FastAPI devuelva error 500 y protege el flujo completo:
    Chat → n8n → FastAPI → proveedor IA.
    """

    print(f"⚠️ Error proveedor externo: {error}")

    return """
{
  "assistant_reply": "Estoy teniendo un inconveniente técnico temporal con el motor de IA. Intentemos nuevamente en unos segundos.",
  "updated_lead_state": {},
  "next_action": "continue_conversation",
  "confidence": 0.0
}
"""


def _call_gemini(prompt: str) -> str:
    """
    Llama a Gemini Developer API.

    Espera que Gemini devuelva JSON como texto usando responseMimeType.
    """

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{EXTERNAL_LLM_MODEL}:generateContent?key={EXTERNAL_LLM_API_KEY}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()

        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return _fallback_response(e)


def _call_openai(prompt: str) -> str:
    """
    Llama a OpenAI Responses API.

    Mantiene una interfaz compatible con el resto del agente:
    recibe prompt y retorna texto JSON.
    """

    try:
        url = "https://api.openai.com/v1/responses"

        headers = {
            "Authorization": f"Bearer {EXTERNAL_LLM_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": EXTERNAL_LLM_MODEL,
            "input": prompt,
            "temperature": 0.2,
            "text": {
                "format": {
                    "type": "json_object"
                }
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()

        data = response.json()
        return data["output_text"]

    except Exception as e:
        return _fallback_response(e)


def _call_mistral(prompt: str) -> str:
    """
    Llama a Mistral Chat Completions API.

    Retorna el contenido del mensaje generado por el modelo.
    """

    try:
        url = "https://api.mistral.ai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {EXTERNAL_LLM_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": EXTERNAL_LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return _fallback_response(e)
    


def _call_openrouter(prompt: str) -> str:
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {EXTERNAL_LLM_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "premium-car-ai-agent",
        }

        payload = {
            "model": EXTERNAL_LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
        }

        start_time = time.perf_counter()

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45,
        )

        elapsed_seconds = time.perf_counter() - start_time

        response.raise_for_status()
        data = response.json()

        debug_llm_response(
            model_configured=EXTERNAL_LLM_MODEL,
            response_json=data,
            elapsed_seconds=elapsed_seconds,
            prompt=prompt,
        )

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return _fallback_response(e)