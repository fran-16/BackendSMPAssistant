from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta la API KEY de OpenAI")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="SMP Assistant API",
    description="Backend de recomendaciones personalizadas (IA) para la app SMP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class MealItem(BaseModel):
    mealType: str     
    id: str
    name: str
    grams: int
    ig: float
    carbs_g: float
    protein_g: float
    fiber_g: float
    kcal: int
    gl: float
    portion_text: str

class DaySummary(BaseModel):
    baseGoal: int
    consumed: int
    remaining: int
    smpCurrent: int

class SuggestionRequest(BaseModel):
    summary: DaySummary
    meals: List[MealItem]
    profile: str
    user_message: Optional[str] = None

class SuggestionResponse(BaseModel):
    suggestion: str



@app.post("/ai/suggestions", response_model=SuggestionResponse)
async def get_suggestions(body: SuggestionRequest):
    """
    Genera una recomendación personalizada en función de:
    - Resumen del día (meta, consumido, resto, SMP)
    - Comidas del día con métricas (IG, GL, carbs, fibra, proteína, kcal)
    - Perfil y mensaje del usuario
    """

    from collections import defaultdict

    meals_by_type = defaultdict(list)
    for m in body.meals:
        meals_by_type[m.mealType].append(m)

    comidas_str_parts = []
    for meal_type, items in meals_by_type.items():
        header = meal_type.capitalize()
        lines = [
            f"  - {it.name} ({it.grams} g, {it.kcal} kcal, IG {it.ig}, GL {it.gl}, "
            f"carbs {it.carbs_g} g, fibra {it.fiber_g} g, prot {it.protein_g} g)"
            for it in items
        ]
        block = header + ":\n" + "\n".join(lines)
        comidas_str_parts.append(block)

    comidas_str = "\n\n".join(comidas_str_parts) if comidas_str_parts else "No hay comidas registradas."

    user_msg = body.user_message or "Dame una recomendación para mejorar mi día hoy."

    prompt = f"""
Eres un asistente especializado en salud metabólica y alimentación.
Tu tarea es dar recomendaciones muy concretas y accionables para hoy.

Perfil del usuario:
- {body.profile}

Resumen del día:
- Meta de calorías (baseGoal): {body.summary.baseGoal} kcal
- Consumido: {body.summary.consumed} kcal
- Restantes: {body.summary.remaining} kcal
- SMP actual del día: {body.summary.smpCurrent} / 100

Comidas del día (con índice glucémico y carga glucémica):
{comidas_str}

Mensaje del usuario:
\"\"\"{user_msg}\"\"\"

Instrucciones:
- Da UNA sola recomendación breve (máximo 2 líneas).
- Ten en cuenta el SMP, la carga glucémica aproximada (GL), el IG y las calorías consumidas vs meta.
- Usa un tono amable, práctico y enfocado en el siguiente paso del día.
- No repitas los datos anteriores, solo da la recomendación.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un coach de salud metabólica breve, muy concreto y práctico."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )

    text = completion.choices[0].message.content.strip()
    return SuggestionResponse(suggestion=text)


@app.get("/")
async def root():
    return {"status": "ok", "message": "SMP Assistant API funcionando 🚀"}
