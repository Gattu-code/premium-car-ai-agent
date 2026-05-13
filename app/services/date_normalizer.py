"""
Módulo: date_normalizer

Responsabilidad:
Normalizar fechas relativas mencionadas por el cliente.

Ejemplos:
- "mañana" -> fecha real en formato ISO
- "próximo martes" -> siguiente martes real
- "este sábado" -> sábado de la semana actual si aún no pasó
- "lunes" -> próximo lunes disponible

Zona horaria:
- America/Bogota
"""

import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


BOGOTA_TZ = ZoneInfo("America/Bogota")


WEEKDAYS = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "miércoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "sábado": 5,
    "domingo": 6,
}


def normalize_text(value: str) -> str:
    """
    Normaliza texto para comparar sin tildes, mayúsculas o espacios extra.
    """
    value = str(value or "").lower().strip()

    value = unicodedata.normalize("NFD", value)
    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Mn"
    )

    value = re.sub(r"\s+", " ", value)

    return value


def get_today(now: Optional[datetime] = None) -> date:
    """
    Retorna la fecha actual en zona horaria de Colombia.
    """
    if now:
        return now.astimezone(BOGOTA_TZ).date()

    return datetime.now(BOGOTA_TZ).date()


def next_weekday(
    today: date,
    target_weekday: int,
    force_next_week: bool = False,
) -> date:
    """
    Calcula la próxima fecha correspondiente a un día de la semana.

    Regla comercial:
    - "martes", "este martes" o "próximo martes" se interpretan como
      el primer martes que sigue.
    - Si hoy es martes, se interpreta como el martes de la siguiente semana.
    """
    days_ahead = target_weekday - today.weekday()

    if days_ahead <= 0:
        days_ahead += 7

    return today + timedelta(days=days_ahead)


def parse_iso_date(text: str) -> Optional[date]:
    """
    Detecta si el texto ya trae una fecha ISO YYYY-MM-DD.
    """
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)

    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_appointment_date(
    value: Any,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Normaliza una fecha de cita escrita por el cliente o por el LLM.

    Returns:
        Dict con:
        - raw
        - iso_date
        - display_date
        - confidence
        - reason
    """
    raw = str(value or "").strip()

    if not raw:
        return {
            "raw": "",
            "iso_date": "",
            "display_date": "",
            "confidence": "low",
            "reason": "No appointment_date provided.",
        }

    today = get_today(now)
    text = normalize_text(raw)

    # Fecha ISO directa
    iso = parse_iso_date(text)
    if iso:
        return {
            "raw": raw,
            "iso_date": iso.isoformat(),
            "display_date": raw,
            "confidence": "high",
            "reason": "ISO date detected.",
        }

    # Fechas relativas simples
    if "hoy" in text:
        target = today
        return {
            "raw": raw,
            "iso_date": target.isoformat(),
            "display_date": raw,
            "confidence": "high",
            "reason": "Relative date: today.",
        }

    if "pasado manana" in text or "pasado mañana" in raw.lower():
        target = today + timedelta(days=2)
        return {
            "raw": raw,
            "iso_date": target.isoformat(),
            "display_date": raw,
            "confidence": "high",
            "reason": "Relative date: day after tomorrow.",
        }

    if "manana" in text or "mañana" in raw.lower():
        target = today + timedelta(days=1)
        return {
            "raw": raw,
            "iso_date": target.isoformat(),
            "display_date": raw,
            "confidence": "high",
            "reason": "Relative date: tomorrow.",
        }

    # Días de la semana
    force_next_week = (
        "proximo" in text
        or "proxima" in text
        or "siguiente" in text
    )

    for day_name, weekday_number in WEEKDAYS.items():
        day_name_norm = normalize_text(day_name)

        if re.search(rf"\b{day_name_norm}\b", text):
            target = next_weekday(
                today=today,
                target_weekday=weekday_number,
                force_next_week=force_next_week,
            )

            return {
                "raw": raw,
                "iso_date": target.isoformat(),
                "display_date": raw,
                "confidence": "medium",
                "reason": f"Weekday detected: {day_name}.",
            }

    return {
        "raw": raw,
        "iso_date": "",
        "display_date": raw,
        "confidence": "low",
        "reason": "Could not normalize appointment date.",
    }


def message_has_relative_date(text: str) -> bool:
    """
    Detecta si un texto contiene una fecha relativa o un día de la semana.
    """
    normalized = normalize_text(text)

    date_terms = [
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
        "proximo",
        "proxima",
        "siguiente",
    ]

    return any(term in normalized for term in date_terms)


def resolve_appointment_date(
    user_message: str,
    current_appointment_date: str,
    now: datetime,
) -> Dict[str, Any]:
    """
    Resuelve la fecha de cita priorizando el mensaje actual del usuario
    cuando contiene una fecha relativa.

    Esto evita conservar una fecha ISO mal calculada por el LLM.
    """
    sources = []

    if message_has_relative_date(user_message):
        sources.append(user_message)

    if current_appointment_date:
        sources.append(current_appointment_date)

    for source in sources:
        normalized = normalize_appointment_date(
            value=source,
            now=now,
        )

        if normalized.get("iso_date"):
            return {
                **normalized,
                "source": source,
            }

    return {
        "raw": "",
        "iso_date": "",
        "display_date": "",
        "confidence": "low",
        "reason": "no_date_detected",
        "source": None,
    }