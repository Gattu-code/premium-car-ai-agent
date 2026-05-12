# Premium Car AI Agent

## Estado actual
Proyecto inicial creado en VS Code con entorno virtual y estructura base.

## Módulo implementado
### `app/models/lead_model.py`
Modelo principal del lead comercial.

### Responsabilidades
- almacenar información del prospecto,
- actualizar datos progresivamente,
- determinar si el lead está listo para ventas,
- calcular prioridad comercial básica.

### Métodos implementados
- `update_from_dict(data)`
- `is_ready_for_sales()`
- `calculate_priority()`
- `refresh_business_state()`

### Prueba realizada
Caso de prueba con:
- `vehicle_segment = SUV`
- `budget_range = 250M COP`
- `purchase_timeframe = inmediata`
- `phone = 3001234567`
- `lead_temperature = caliente`

### Resultado esperado
- `qualified = True`
- `priority_score = 90`


### `app/services/ollama_service.py`

Servicio encargado de la comunicación con el modelo de IA local (Ollama).

#### Responsabilidades
- enviar prompts al modelo LLM,
- recibir respuestas generadas,
- manejar errores de conexión y formato.

#### Función principal
`generate_with_ollama(prompt: str) -> str`

#### Requisitos
- Ollama debe estar corriendo en:
  http://localhost:11434
- Modelo descargado:
  qwen2.5:3b

#### Prueba manual
Ejecutar en terminal:

```bash
ollama run qwen2.5:3b


### `app/prompts/premium_prompt.txt`

Define el comportamiento del agente de ventas premium.

#### Responsabilidades
- controlar tono conversacional,
- definir reglas de interacción,
- estructurar salida JSON,
- guiar extracción de datos del lead.

#### Características clave
- tono premium y consultivo,
- máximo 1-2 preguntas por respuesta,
- enfoque en perfilamiento progresivo,
- clasificación de leads (frío, tibio, caliente),
- control de acciones del flujo.

#### Output esperado
Siempre JSON válido con:
- assistant_reply
- updated_lead_state
- next_action
- confidence


### `app/agents/premium_agent.py`

Módulo orquestador del agente comercial.

#### Responsabilidades
- construir el prompt final,
- combinar historial, lead y contexto comercial,
- llamar a Ollama,
- parsear la salida JSON,
- actualizar el lead,
- recalcular prioridad y calificación.

#### Funciones principales
- `load_base_prompt()`
- `build_prompt(...)`
- `safe_parse_json(...)`
- `normalize_agent_response(...)`
- `process_lead_message(...)`

#### Regla importante
Aunque la IA sugiera una acción, el backend recalcula el estado comercial real del lead y puede ajustar:
- `send_to_sales_advisor`
- `high_priority_lead`


### `app/main.py`

Punto de entrada de la API con FastAPI.

#### Responsabilidades
- publicar endpoints del sistema,
- recibir requests del agente,
- delegar la lógica al módulo `premium_agent`.

#### Endpoints implementados

##### `GET /health`
Valida que la API esté operativa.

##### `GET /lead-state/empty`
Retorna la estructura vacía oficial del lead.

##### `POST /premium-agent`
Procesa un mensaje del usuario usando:
- estado del lead,
- historial conversacional,
- contexto comercial opcional.

### `app/services/lead_normalizer.py`

Servicio encargado de normalizar datos del lead para consistencia de negocio.

#### Filosofía
- La IA interpreta lenguaje natural
- El backend estandariza los datos

#### Responsabilidades actuales
- normalizar presupuesto a formato colombiano (COP)

#### Ejemplos
- "250m" → "250 millones COP"
- "250000000" → "250 millones COP"

#### Futuras mejoras
- normalización de teléfonos
- validación de emails
- estandarización de ciudades

### Mejora del prompt comercial

Se reforzó `premium_prompt.txt` para reducir errores de interpretación en contexto colombiano.

#### Mejoras introducidas
- contexto explícito de Colombia,
- reglas monetarias para expresiones como `250 millones`, `180m`, `300 palos`,
- prohibición de conversión automática a USD,
- prohibición de inventar rangos de presupuesto,
- reglas más estrictas de extracción fiel.

#### Objetivo
Reducir errores del modelo antes de aumentar complejidad en backend.

## Enrutamiento comercial del lead

Se agregó una capa backend para decidir la siguiente acción comercial
sin depender completamente de la IA.

### Archivo
`app/services/lead_router.py`

### Acciones posibles
- `continue_conversation`
- `request_contact_info`
- `send_to_sales_advisor`
- `high_priority_lead`

### Objetivo
Aplicar reglas comerciales consistentes sobre el estado del lead.

## Alineación entre acción y respuesta comercial

Cuando el backend determina que la siguiente acción es `request_contact_info`,
la respuesta del agente se ajusta para pedir datos de contacto de forma coherente
con el flujo comercial.

### Objetivo
Evitar que el sistema marque `request_contact_info` mientras el texto todavía
hace preguntas exploratorias no alineadas con el cierre.