"""
Módulo: contact_validation_service

Responsabilidad:
Validar y normalizar datos básicos de contacto del lead:
- teléfono
- correo electrónico

Estas validaciones evitan confirmar citas con datos incompletos o inválidos.
"""

import re


def normalize_email(value: str) -> str:
    """
    Normaliza email básico.
    """
    return str(value or "").strip().lower()


def normalize_phone(value: str) -> str:
    """
    Conserva solo dígitos del teléfono.
    Ejemplo:
    "(300) 123-4567" -> "3001234567"
    """
    return re.sub(r"\D", "", str(value or ""))


def is_valid_email(value: str) -> bool:
    """
    Valida formato básico de email.
    Requiere:
    - texto antes de @
    - dominio
    - punto en el dominio
    """
    email = normalize_email(value)

    if not email:
        return False

    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    return bool(re.match(pattern, email))


def is_valid_phone(value: str) -> bool:
    """
    Valida teléfono básico.

    Regla MVP:
    - solo dígitos después de normalizar
    - mínimo 7 dígitos
    - máximo 15 dígitos
    """
    phone = normalize_phone(value)

    return 7 <= len(phone) <= 15


def normalize_contact_fields(lead):
    """
    Normaliza teléfono y correo directamente sobre el lead.
    """
    if getattr(lead, "email", None):
        lead.email = normalize_email(lead.email)

    if getattr(lead, "phone", None):
        lead.phone = normalize_phone(lead.phone)

    return lead


def get_invalid_contact_fields(lead) -> list[str]:
    """
    Retorna campos de contacto inválidos.
    Solo marca inválido si el campo existe pero no cumple formato.
    """
    invalid_fields = []

    if getattr(lead, "phone", None) and not is_valid_phone(lead.phone):
        invalid_fields.append("teléfono")

    if getattr(lead, "email", None) and not is_valid_email(lead.email):
        invalid_fields.append("correo electrónico")

    return invalid_fields