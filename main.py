from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv  

load_dotenv()

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

    meals_by_type = {}
    for m in body.meals:
        meals_by_type.setdefault(m.mealType, []).append(m)

    comidas_str_parts = []
    for meal_type, items in meals_by_type.items():
        block = meal_type.capitalize() + ":\n" + "\n".join([
            f"  - {it.name} ({it.grams} g, {it.kcal} kcal, IG {it.ig}, GL {it.gl}, "
            f"carbs {it.carbs_g} g, fibra {it.fiber_g} g, prot {it.protein_g} g)"
            for it in items
        ])
        comidas_str_parts.append(block)

    comidas_str = "\n\n".join(comidas_str_parts) if comidas_str_parts else "No hay comidas registradas."

    user_msg = body.user_message or "Dame una recomendación personalizada."

    prompt = f"""
Perfil del usuario:
{body.profile}

Resumen del día:
- Meta: {body.summary.baseGoal} kcal
- Consumidas: {body.summary.consumed}
- Restantes: {body.summary.remaining}
- SMP: {body.summary.smpCurrent}

Comidas:
{comidas_str}

Mensaje del usuario:
{user_msg}

 DA UNA SOLA RECOMENDACIÓN clara, útil, breve (máximo 2 líneas) y aplicable hoy.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un coach de salud metabólica conciso y muy práctico."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
    )

    text = completion.choices[0].message.content.strip()
    return SuggestionResponse(suggestion=text)


@app.get("/")
async def root():
    return {"status": "ok", "message": "SMP Assistant API funcionando "}
