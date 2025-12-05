from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from openai import OpenAI
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta la API KEY de OpenAI")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="SMP Assistant API",
    description="Backend de recomendaciones personalizadas (IA) para la app SMP",
    version="1.1.0",
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



class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    summary: DaySummary
    meals: List[MealItem]
    profile: str
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str



def build_meal_context(meals: List[MealItem]):
    from collections import defaultdict
    meals_by_type = defaultdict(list)

    for m in meals:
        meals_by_type[m.mealType].append(m)

    blocks = []
    for meal_type, items in meals_by_type.items():
        lines = [
            f"  - {i.name} ({i.grams} g, {i.kcal} kcal, IG {i.ig}, GL {i.gl}, "
            f"carbs {i.carbs_g} g, fibra {i.fiber_g} g, prot {i.protein_g} g)"
            for i in items
        ]
        blocks.append(meal_type.capitalize() + ":\n" + "\n".join(lines))

    return "\n\n".join(blocks) if blocks else "No hay comidas registradas."


@app.post("/ai/suggestions", response_model=SuggestionResponse)
async def get_suggestions(body: SuggestionRequest):

    comidas_str = build_meal_context(body.meals)
    user_msg = body.user_message or "Dame una recomendación para mejorar mi salud metabólica hoy."

    prompt = f"""
Eres un experto en salud metabólica y nutrición personalizada.
Ten muy en cuenta las enfermedades y las restricciones alimentarias del usuario.

PERFIL DEL USUARIO:
{body.profile}


OBJETIVO DEL DÍA:
Meta: {body.summary.baseGoal} kcal
Consumido: {body.summary.consumed}
Restantes: {body.summary.remaining}
SMP actual: {body.summary.smpCurrent}

COMIDAS DEL DÍA:
{comidas_str}

MENSAJE DEL USUARIO:
\"\"\"{user_msg}\"\"\"

Da UNA sola recomendación clara, segura y personalizada para hoy.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )

    return SuggestionResponse(
        suggestion=completion.choices[0].message.content.strip()
    )


@app.post("/ai/chat", response_model=ChatResponse)
async def chat_with_bot(body: ChatRequest):

    comidas_str = build_meal_context(body.meals)

    history_str = ""
    for msg in body.messages:
        speaker = "Usuario" if msg.role == "user" else "Coach"
        history_str += f"{speaker}: {msg.content}\n"

    last_user_message = next(
        (m.content for m in reversed(body.messages) if m.role == "user"),
        "Necesito ayuda para mejorar hoy."
    )

    system_prompt = """
Eres un nutricionista y coach especializado en salud metabólica y resistencia a la insulina.
Reglas:
- Sé amable, motivador, práctico.
- Máximo 3-4 frases por respuesta.
- Usa el contexto metabólico para dar consejos útiles.
- No hagas diagnósticos médicos.
"""

    user_prompt = f"""
Contexto metabólico del usuario:

Meta kcal: {body.summary.baseGoal}
Consumidas: {body.summary.consumed}
Restantes: {body.summary.remaining}
SMP actual: {body.summary.smpCurrent}/100

Perfil:
{body.profile}

Comidas del día:
{comidas_str}

Conversación previa:
{history_str}

Última duda del usuario:
\"\"\"{last_user_message}\"\"\"

Da la mejor respuesta posible.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.65,
    )

    return ChatResponse(
        reply=completion.choices[0].message.content.strip()
    )



@app.get("/")
async def root():
    return {"status": "ok", "message": "SMP Assistant API funcionando"}

