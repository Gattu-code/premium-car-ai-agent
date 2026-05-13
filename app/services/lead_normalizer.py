"""
Módulo: lead_normalizer

Responsabilidad:
Aplicar reglas de negocio sobre el estado del lead para garantizar
consistencia en los datos antes de usarlos en procesos comerciales
(CRM, scoring, analítica, etc).

Filosofía:
- La IA interpreta lenguaje natural (flexible).
- El backend normaliza y estandariza (determinístico).

Este módulo:
- NO intenta entender lenguaje complejo (eso lo hace la IA),
- SÍ convierte valores ambiguos a formatos consistentes.

Actualmente implementa:
- normalización de presupuesto en contexto colombiano (COP).
"""

import re
from typing import Optional

from app.models.lead_model import LeadModel
from app.services.date_normalizer import normalize_appointment_date
from app.services.contact_validation_service import normalize_contact_fields
from app.services.date_normalizer import normalize_text


def clean_lead_name_if_derived_from_email(lead):
    if not getattr(lead, "lead_name", None):
        return lead

    if not getattr(lead, "email", None):
        return lead

    email_user = str(lead.email).split("@")[0].strip().lower()
    lead_name = str(lead.lead_name).strip().lower()

    if lead_name == email_user:
        lead.lead_name = ""

    return lead

# =========================
# NORMALIZACIÓN DE PRESUPUESTO
# =========================

def normalize_budget_range(raw_budget: Optional[str]) -> Optional[str]:
    """
    Normaliza el presupuesto del lead a un formato estándar colombiano.

    Objetivo:
    Convertir múltiples formas de expresar dinero en un formato consistente.

    Ejemplos de entrada:
        "250 millones"
        "250m"
        "250 M COP"
        "cop 250000000"
        "250000000"
        "$250,000"  (caso ambiguo, se deja igual por ahora)

    Ejemplo de salida:
        "250 millones COP"

    Reglas:
    - Si detecta "millones" o "m", lo interpreta como millones COP.
    - Si detecta números grandes (>= 8 dígitos), los convierte a millones.
    - Si no puede interpretar con seguridad, devuelve el valor original.

    Args:
        raw_budget (str | None):
            Valor original del presupuesto generado por la IA.

    Returns:
        str | None:
            Presupuesto normalizado o valor original si no se puede transformar.
    """
    if not raw_budget:
        return raw_budget

    text = raw_budget.strip().lower()

    # -------------------------
    # Caso 1: "250 millones", "250m", "250 millón"
    # -------------------------
    match_millions = re.search(r"(\d+(?:[.,]\d+)?)\s*(millones|millon|millón|m)\b", text)
    if match_millions:
        number = match_millions.group(1).replace(",", ".")
        return f"{number} millones COP"

    # -------------------------
    # Caso 2: números grandes (ej: 250000000)
    # -------------------------
    match_large_number = re.search(r"\b(\d{8,12})\b", text)
    if match_large_number:
        value = match_large_number.group(1)

        try:
            numeric_value = int(value)
            millions = numeric_value / 1_000_000

            # Si es entero, quitar decimales
            if millions.is_integer():
                millions = int(millions)

            return f"{millions} millones COP"

        except ValueError:
            return raw_budget

    # -------------------------
    # Caso 3: ya viene en COP explícito
    # -------------------------
    if "cop" in text and "millones" in text:
        return raw_budget

    # -------------------------
    # Caso por defecto (no seguro)
    # -------------------------
    return raw_budget

def clean_purchase_timeframe_if_appointment_date(lead):
    """
    Limpia purchase_timeframe cuando el LLM guardó allí una fecha de visita.

    Ejemplo incorrecto:
    - purchase_timeframe = "este sábado"
    - appointment_date = "2026-05-16"

    En ese caso, "este sábado" corresponde a la cita, no a la intención de compra.
    """

    if not getattr(lead, "purchase_timeframe", None):
        return lead

    if not getattr(lead, "appointment_date", None):
        return lead

    normalized_timeframe = normalize_text(lead.purchase_timeframe)

    appointment_date_terms = [
        "hoy",
        "manana",
        "pasado manana",
        "lunes",
        "martes",
        "miercoles",
        "jueves",
        "viernes",
        "sabado",
        "domingo",
        "este lunes",
        "este martes",
        "este miercoles",
        "este jueves",
        "este viernes",
        "este sabado",
        "este domingo",
        "proximo lunes",
        "proximo martes",
        "proximo miercoles",
        "proximo jueves",
        "proximo viernes",
        "proximo sabado",
        "proximo domingo",
    ]

    if normalized_timeframe in appointment_date_terms:
        lead.purchase_timeframe = ""

    return lead

# =========================
# NORMALIZACIÓN COMPLETA DEL LEAD
# =========================

def normalize_lead(lead: LeadModel) -> LeadModel:
    """
    Aplica todas las reglas de normalización al lead.

    Este método es el punto único donde se aplican
    transformaciones de datos de negocio.

    Actualmente:
    - normaliza el presupuesto

    Futuro:
    - normalización de teléfonos
    - estandarización de ciudades
    - limpieza de nombres
    - validación de emails

    Args:
        lead (LeadModel):
            Objeto de lead a normalizar.

    Returns:
        LeadModel:
            Lead con valores estandarizados.
    """
    lead = normalize_contact_fields(lead)        
    lead = clean_lead_name_if_derived_from_email(lead)
    
    # Normalizar presupuesto
    lead.budget_range = normalize_budget_range(lead.budget_range)
    
    # -----------------------------
    # Normalización de fecha de cita
    # -----------------------------
    # appointment_date conserva el texto conversacional.
    # appointment_date_iso guarda la fecha real normalizada.
    if getattr(lead, "appointment_date", None):
        normalized_date = normalize_appointment_date(
            lead.appointment_date
        )

        if normalized_date.get("iso_date"):
            lead.appointment_date_iso = normalized_date["iso_date"]
    
    lead = clean_purchase_timeframe_if_appointment_date(lead)
    
    return lead