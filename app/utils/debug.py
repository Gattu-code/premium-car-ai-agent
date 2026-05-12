"""
Módulo: debug

Responsabilidad:
Centralizar logs de debugging del sistema.

Permite activar/desactivar logs sin contaminar la lógica de negocio.
"""

import time

# 🔥 Toggle global
DEBUG_ENABLED = True


def debug_print(message: str):
    """
    Imprime mensaje solo si debug está activado.
    """
    if DEBUG_ENABLED:
        print(message)


def debug_prompt(prompt: str):
    """
    Log del tamaño y preview del prompt.
    """
    if not DEBUG_ENABLED:
        return

    print(f"[DEBUG] Prompt length: {len(prompt)}")
    #print("[DEBUG] Prompt preview:")
    #print(prompt[:1000])


def debug_timer_start() -> float:
    """
    Inicia medición de tiempo.
    """
    if not DEBUG_ENABLED:
        return 0.0

    return time.time()


def debug_timer_end(start: float, label: str = "Operation"):
    """
    Finaliza medición de tiempo.
    """
    if not DEBUG_ENABLED:
        return

    elapsed = time.time() - start
    print(f"[DEBUG] {label} took {elapsed:.2f}s")
    
def debug_llm_response(
        model_configured: str,
        response_json: dict,
        elapsed_seconds: float,
        prompt: str = "",
):
    """
    Log de métricas de la llamada al LLM.

    Incluye:
    - modelo configurado,
    - modelo reportado por el proveedor,
    - tiempo de respuesta,
    - caracteres del prompt,
    - tokens reales si el proveedor los devuelve,
    - costo si viene disponible.
    """
    if not DEBUG_ENABLED:
        return

    usage = response_json.get("usage", {}) or {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    cost = usage.get("cost")

    estimated_prompt_tokens = round(len(prompt) / 4) if prompt else None

    print("\n" + "=" * 60)
    print("[DEBUG] LLM CALL")
    print("=" * 60)
    print(f"[DEBUG] Modelo configurado: {model_configured}")
    print(f"[DEBUG] Modelo reportado: {response_json.get('model', 'No reportado')}")
    print(f"[DEBUG] Generation ID: {response_json.get('id', 'No reportado')}")
    print(f"[DEBUG] Tiempo LLM: {elapsed_seconds:.2f}s")

    if prompt:
        print(f"[DEBUG] Prompt chars: {len(prompt)}")
        print(f"[DEBUG] Prompt tokens estimados: {estimated_prompt_tokens}")

    print("-" * 60)
    print("[DEBUG] USAGE")
    print(f"[DEBUG] Prompt tokens: {prompt_tokens}")
    print(f"[DEBUG] Completion tokens: {completion_tokens}")
    print(f"[DEBUG] Total tokens: {total_tokens}")
    print(f"[DEBUG] Cost: {cost}")
    print("=" * 60 + "\n")