"""
Configuración central del proyecto.

Aquí se define qué motor de IA usará el agente.

Modos:
- AI_PROVIDER = "ollama": usa modelo local vía Ollama.
- AI_PROVIDER = "external": usa una API externa.

Para APIs externas:
- EXTERNAL_LLM_PROVIDER define el proveedor específico.
- EXTERNAL_LLM_API_KEY se lee desde variable de entorno.
- EXTERNAL_LLM_MODEL define el modelo.
"""

import os

CATALOG_BRAND = "volvo"
CATALOG_MARKET = "colombia"
CATALOG_PATH = "data/catalogs/volvo_colombia.json"

# =========================
# AI Provider
# =========================

# AI_PROVIDER = "ollama"
AI_PROVIDER = "external"

# =========================
# External LLM Provider
# =========================

# EXTERNAL_LLM_PROVIDER = "gemini"
# EXTERNAL_LLM_PROVIDER = "openai"
# EXTERNAL_LLM_PROVIDER = "mistral"
EXTERNAL_LLM_PROVIDER = "openrouter"

# =========================
# Model options
# Para cambiar modelo en producción, cambiar EXTERNAL_LLM_MODEL en .env
# =========================

# Opciones probadas:
# google/gemini-2.5-flash-lite
# google/gemini-2.5-flash
# deepseek/deepseek-v3.2
# openai/gpt-4o-mini

EXTERNAL_LLM_MODEL = os.getenv(
    "EXTERNAL_LLM_MODEL",
    "google/gemini-2.5-flash-lite"
)

# =========================
# API Keys / URLs
# =========================

EXTERNAL_LLM_API_KEY = os.getenv("EXTERNAL_LLM_API_KEY")

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)