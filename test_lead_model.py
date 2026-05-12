from app.models.lead_model import LeadModel

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