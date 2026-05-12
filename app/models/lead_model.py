from pydantic import BaseModel, Field
from typing import List, Optional


class LeadModel(BaseModel):
    """
    Modelo principal del lead para el agente de vehículos premium.

    Este modelo representa el estado acumulado de un prospecto durante
    la conversación con la IA. Su propósito es:
    1. almacenar la información del lead,
    2. permitir actualizaciones progresivas,
    3. determinar si ya está listo para pasar a ventas,
    4. calcular una prioridad comercial básica.
    """

    # =========================
    # Información básica
    # =========================
    lead_name: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    city: Optional[str] = ""

    # =========================
    # Interés en vehículo
    # =========================
    brand: Optional[str] = "" #marca ej, volvo, fiat, jeep, ....
    vehicle_interest: Optional[str] = ""   # Ej: XC60, BMW X3
    vehicle_segment: Optional[str] = ""    # SUV, sedán, eléctrico

    # =========================
    # Capacidad de compra
    # =========================
    budget_range: Optional[str] = ""
    payment_method: Optional[str] = ""     # contado, financiación, leasing
    purchase_timeframe: Optional[str] = "" # inmediata, 1 mes, explorando

    # =========================
    # Contexto del cliente
    # =========================
    current_vehicle: Optional[str] = ""
    has_trade_in: Optional[bool] = None
    lifestyle_profile: Optional[str] = ""  # ejecutivo, familia, etc.
    family_context: Optional[str] = ""
    primary_motivation: Optional[str] = "" # seguridad, lujo, tecnología

    # =========================
    # Clasificación del lead
    # =========================
    interest_type: Optional[str] = ""      # exploración, comparación, decisión
    lead_temperature: Optional[str] = ""   # frío, tibio, caliente
    priority_score: int = 0                # rango 0 - 100

    # =========================
    # Estado del flujo
    # =========================
    qualified: bool = False
    pending_questions: List[str] = Field(default_factory=list)

    # =========================
    # Resumen final
    # =========================
    summary: Optional[str] = ""

    # =========================
    # Cita / test drive
    # =========================
    appointment_date: Optional[str] = ""
    appointment_time: Optional[str] = ""
    appointment_location: Optional[dict] = None
    appointment_date_iso: Optional[str] = None
    
    # =========================
    # MÉTODOS DE NEGOCIO
    # =========================

    def update_from_dict(self, data: dict) -> None:
        """
        Actualiza el lead con datos nuevos sin borrar información útil existente.

        Reglas:
        - solo actualiza atributos que existan en el modelo,
        - ignora valores None o cadenas vacías,
        - permite que la IA vaya enriqueciendo el lead paso a paso.

        Args:
            data (dict): diccionario con posibles campos a actualizar.
        """
        for key, value in data.items():
            if hasattr(self, key) and value not in [None, ""]:
                setattr(self, key, value)

    def is_ready_for_sales(self) -> bool:
        """
        Determina si el lead ya tiene información mínima para ser derivado
        a un asesor comercial humano.

        Criterios mínimos actuales:
        - segmento de vehículo,
        - rango de presupuesto,
        - horizonte de compra,
        - teléfono de contacto.from app.models.lead_model import LeadModel

lead = LeadModel()

lead.update_from_dict({
    "vehicle_segment": "SUV",
    "budget_range": "250M COP",
    "purchase_timeframe": "inmediata",
    "phone": "3001234567",
    "lead_temperature": "caliente"
})

lead.refresh_business_state()

print(lead)
print("Qualified:", lead.qualified)
print("Priority:", lead.priority_score)

        Returns:
            bool: True si el lead ya está listo para ventas, False en caso contrario.
        """
        required_fields = [
            self.vehicle_segment,
            self.budget_range,
            self.purchase_timeframe,
            self.phone,
        ]
        return all(required_fields)

    def calculate_priority(self) -> None:
        """
        Calcula un puntaje simple de prioridad comercial del lead.

        Reglas actuales:
        - presupuesto definido: +20
        - compra inmediata o en 1 mes: +30
        - método de pago definido: +10
        - teléfono disponible: +20
        - lead caliente: +20

        El puntaje máximo es 100.

        Este método puede evolucionar después con reglas más sofisticadas.
        """
        score = 0

        if self.budget_range:
            score += 20

        if self.purchase_timeframe in ["inmediata", "1 mes"]:
            score += 30

        if self.payment_method:
            score += 10

        if self.phone:
            score += 20

        if self.lead_temperature == "caliente":
            score += 20

        self.priority_score = min(score, 100)

    def refresh_business_state(self):
        """
        Recalcula el estado comercial del lead.

        Responsabilidad:
        - Calcular score base
        - Determinar temperatura
        - Determinar si está calificado
        """

        score = 0

        has_name = bool(self.lead_name)
        has_city = bool(self.city)
        has_model = bool(self.vehicle_interest)
        has_budget = bool(self.budget_range)
        has_payment = bool(self.payment_method)
        has_phone = bool(self.phone)
        has_email = bool(self.email)
        has_date = bool(self.appointment_date)
        has_time = bool(self.appointment_time)

        # -----------------------------
        # Score base
        # -----------------------------
        if has_name:
            score += 5

        if has_city:
            score += 10

        if has_model:
            score += 15

        if has_budget:
            score += 10

        if has_payment:
            score += 10

        if has_phone:
            score += 20

        if has_email:
            score += 10

        if has_date:
            score += 10

        if has_time:
            score += 10

        score = min(score, 100)

        self.priority_score = score

        # -----------------------------
        # Clasificación del lead
        # -----------------------------
        if all([has_phone, has_model, has_city, has_date, has_time]):
            self.lead_temperature = "caliente"
            self.qualified = True

        elif has_phone and has_model:
            self.lead_temperature = "tibio"
            self.qualified = False

        elif score >= 50:
            self.lead_temperature = "tibio"
            self.qualified = False

        else:
            self.lead_temperature = "frio"
            self.qualified = False