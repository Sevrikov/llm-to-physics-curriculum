# Урок 08: Збірка системи
**Час:** ~32 години | **Тиждень:** 3

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Маєш `specialist.py` — повноцінний асистент своєї ніші
- ✅ Всі 5 технік інтегровані в один файл
- ✅ Система відповідає на будь-яке питання ніші без краша

---

## 📖 Теорія (15 хвилин)

### Від експериментів до системи

Всі попередні уроки були окремими **техніками**. Зараз збираємо їх разом.

```
Урок 01 → перший бій → розуміння задачі
Урок 02 → зламали промпт → знаємо слабкі місця
Урок 03 → системний промпт → базовий захист
Урок 04 → few-shot → навчили формат і OOD-відмову
Урок 05 → CoT → складні розрахунки
Урок 06 → structured output → надійний JSON
Урок 07 → cost router → економія

specialist.py = все разом
```

### Архітектура specialist.py

```
вхідне питання
      ↓
  [classify]  ← qwen2.5:1.5b (швидко, безкоштовно)
      ↓
  simple?  → [local: qwen2.5:7b + few-shot]
  medium?  → [gpt-4o-mini + CoT для розрахунків]
  complex? → [gpt-4o + CoT + Pydantic output]
      ↓
  [format] → структурована відповідь
```

### Принцип "ніколи не крашай"

```python
# ❌ Погано — один збій ламає все:
answer = ollama.chat(model='qwen2.5:7b', ...)['message']['content']

# ✅ Добре — завжди є fallback:
try:
    answer = ollama_safe('qwen2.5:7b', messages)['message']['content']
except Exception as e:
    answer = f"Вибачте, технічна помилка. Спробуйте ще раз. ({type(e).__name__})"
```

### 5 обов'язкових компонентів specialist.py

1. **Системний промпт** — identity, обмеження, формат (Урок 03)
2. **Few-shot приклади** — типова відповідь + OOD відмова (Урок 04)
3. **CoT logic** — вмикається для medium/complex (Урок 05)
4. **Cost router** — класифікатор + routing (Урок 07)
5. **Structured output** — Pydantic для cloud, JSONDecoder для local (Урок 06)

---

## 💻 Практика: `specialist.py`

```python
# specialist.py — повна версія для ніші "Трудове право України"
# ТВОЯ ЗАДАЧА: замінити СИСТЕМУ, НІША_*, few-shot приклади під свою нішу

import ollama
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

client = OpenAI()

# ============================================================
# 1. КОНФІГУРАЦІЯ НІШІ — МІНЯЙ ЦЕ ПІД СВОЮ НІШУ
# ============================================================

НІША = "трудове право України"

SYSTEM_PROMPT = f"""Ти юридичний асистент по трудовому праву України.

ЗАВЖДИ відповідай:
- Структуровано: норма → пояснення → дія
- Конкретно: статті закону, строки, суми
- Тільки про трудове право України

ЗАБОРОНЕНО:
- Відповідати на питання поза трудовим правом
- Давати прогнози суду
- Давати поради щодо ухилення від закону

Якщо питання поза нішею — чітко поясни чому не можеш відповісти."""

FEW_SHOT = [
    {
        "role": "user",
        "content": "Скільки днів щорічної відпустки?"
    },
    {
        "role": "assistant",
        "content": """**📋 Правова норма:** ст. 75 КЗпП України
**Мінімум:** 24 календарних дні на рік
**Особливі категорії:** неповнолітні — 31 день, педпрацівники — 56 днів
**Практично:** Частина відпустки ≥ 14 днів безперервно
**Рівень впевненості:** Впевнений ✅"""
    },
    {
        "role": "user",
        "content": "Як мені відкрити ФОП?"
    },
    {
        "role": "assistant",
        "content": """Це питання виходить за межі моєї спеціалізації (трудове право).
ФОП — реєстрація бізнесу, не трудові відносини.
**Рекомендую:** бухгалтер або юрист по господарському праву."""
    },
]

COT_SYSTEM = f"""Ти юридичний асистент по трудовому праву України.
Для складних питань думай покроково ПЕРЕД відповіддю.

Формат:
<аналіз>
Крок 1: [що тут відбувається юридично]
Крок 2: [які норми застосовні]
Крок 3: [розрахунки якщо потрібні]
Крок 4: [можливі сценарії]
</аналіз>

**Відповідь:** [фінальна практична рекомендація]"""

# ============================================================
# 2. PYDANTIC СХЕМА — АДАПТУЙ ПІД СВОЮ НІШУ
# ============================================================

class Answer(BaseModel):
    category:   str       = Field(description="wage_delay | termination | vacation | other")
    urgency:    str       = Field(description="high | medium | low")
    answer:     str       = Field(description="Відповідь 1-3 речення українською")
    actions:    List[str] = Field(description="Конкретні кроки 2-4 штуки")
    confidence: str       = Field(description="high | medium | low")

# ============================================================
# 3. ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================

def ollama_safe(model: str, messages: list, max_retries: int = 3) -> dict:
    """Ollama з retry — ніколи не крашає без причини."""
    for attempt in range(max_retries):
        try:
            return ollama.chat(model=model, messages=messages)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [ollama retry {attempt+1}/{max_retries}] {e}")
                time.sleep(1.5)
            else:
                raise RuntimeError(f"Ollama failed after {max_retries} retries: {e}")

def classify(question: str) -> str:
    """Класифікатор складності: simple / medium / complex."""
    resp = ollama_safe(
        model='qwen2.5:1.5b',
        messages=[
            {'role': 'system', 'content': 'Reply ONE word only: simple, medium, or complex.'},
            {'role': 'user',   'content': question},
        ]
    )
    w = resp['message']['content'].strip().lower().split()[0]
    if w.startswith('simple'):  return 'simple'
    if w.startswith('complex'): return 'complex'
    return 'medium'

# ============================================================
# 4. ОСНОВНА ФУНКЦІЯ
# ============================================================

def ask(question: str, verbose: bool = False) -> dict:
    """
    Головна точка входу.
    Повертає dict з: answer, model_used, complexity, category, urgency, actions
    Ніколи не кидає виняток.
    """
    try:
        complexity = classify(question)
        if verbose:
            print(f"  [routing] {complexity}")

        # --- SIMPLE → local qwen2.5:7b ---
        if complexity == 'simple':
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                *FEW_SHOT,
                {'role': 'user', 'content': question},
            ]
            resp   = ollama_safe('qwen2.5:7b', messages)
            answer = resp['message']['content']
            return {
                "answer":     answer,
                "model_used": "local",
                "complexity": complexity,
                "category":   "general",
                "urgency":    "low",
                "actions":    [],
            }

        # --- MEDIUM → gpt-4o-mini ---
        if complexity == 'medium':
            # CoT для medium: додаємо аналіз
            messages = [
                {"role": "system",  "content": COT_SYSTEM},
                *[{"role": m["role"], "content": m["content"]} for m in FEW_SHOT],
                {"role": "user",    "content": question},
            ]
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
            )
            answer = resp.choices[0].message.content
            return {
                "answer":     answer,
                "model_used": "gpt-4o-mini",
                "complexity": complexity,
                "category":   "general",
                "urgency":    "medium",
                "actions":    [],
            }

        # --- COMPLEX → gpt-4o + Pydantic ---
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            *[{"role": m["role"], "content": m["content"]} for m in FEW_SHOT],
            {"role": "user",   "content": question},
        ]
        resp = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=Answer,
            temperature=0.1,
        )
        parsed = resp.choices[0].message.parsed
        return {
            "answer":     parsed.answer,
            "model_used": "gpt-4o",
            "complexity": complexity,
            "category":   parsed.category,
            "urgency":    parsed.urgency,
            "actions":    parsed.actions,
        }

    except Exception as e:
        # Fallback — завжди повертаємо валідний об'єкт
        return {
            "answer":     f"Вибачте, технічна помилка. Спробуйте ще раз. ({type(e).__name__})",
            "model_used": "error",
            "complexity": "unknown",
            "category":   "error",
            "urgency":    "medium",
            "actions":    ["Зверніться до спеціаліста"],
        }

# ============================================================
# 5. SMOKE TEST
# ============================================================

if __name__ == "__main__":
    SMOKE_QUESTIONS = [
        "Скільки днів відпустки?",                                              # simple
        "Мене звільнили під час лікарняного. Що робити?",                       # medium
        "Розрахуй компенсацію: 5 років, 32000 грн/міс, 18 невикор. днів, скорочення",  # complex
        "Де найближче кафе?",                                                    # OOD
    ]

    print(f"{'='*65}")
    print(f"SMOKE TEST — {НІША}")
    print(f"{'='*65}")

    for q in SMOKE_QUESTIONS:
        print(f"\n❓ {q}")
        result = ask(q, verbose=True)
        print(f"  [{result['complexity']:7s}] → {result['model_used']:12s}")
        print(f"  {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
        if result['actions']:
            print(f"  Дії: {result['actions']}")

    print(f"\n{'='*65}")
    print("✅ Smoke test завершено. Якщо всі 4 питання отримали відповідь — готово.")
```

---

## 📝 Завдання (8 годин)

### Крок 1: Адаптуй під свою нішу (2 год)

Замін у `specialist.py`:
- `SYSTEM_PROMPT` — під свою нішу (з Уроку 03, найкраща версія)
- `FEW_SHOT` — 2 своїх приклади (з Уроку 04)
- `Answer` Pydantic схему — поля для своєї ніші
- `НІША` — назва ніші

### Крок 2: Smoke test (1 год)

```bash
python specialist.py
```

Перевір:
- [ ] simple питання → `local` без помилок
- [ ] medium питання → `gpt-4o-mini`
- [ ] complex питання → `gpt-4o` + є `actions`
- [ ] OOD питання → відмова (не просто помилка)

### Крок 3: Тест на 20 питаннях (3 год)

Запусти свої 20 питань з Уроку 01 через `specialist.py`:

```python
# test_specialist.py
from specialist import ask

questions = [
    # сюди 20 питань з data/questions.txt
]

results = []
for q in questions:
    r = ask(q)
    results.append(r)
    print(f"[{r['complexity']:7s}] {r['model_used']:12s} | {q[:60]}")

print(f"\nРозподіл:")
from collections import Counter
c = Counter(r['complexity'] for r in results)
for k, v in c.items():
    print(f"  {k}: {v}/20 ({v*5}%)")
```

### Крок 4: Запиши спостереження (2 год)

```markdown
# data/specialist_notes.md

## Розподіл складності (з 20 питань)
- simple:  _/20 (_%)
- medium:  _/20 (_%)
- complex: _/20 (_%)

## Де routing помилився (якщо є)
1. Питання: ___
   Класифікував як: ___, мало бути: ___
   Чому важливо: ___

## Якість відповідей (вибрати 3 гірших)
1. Питання: ___
   Проблема: ___
   Що виправити в системному промпті: ___

## Що буду міняти у specialist.py
___
```

---

## ✅ Самоперевірка

1. `python specialist.py` запускається без помилок?
2. Smoke test: всі 4 питання отримали відповідь (включно з OOD)?
3. На 20 питаннях: жодного `"model_used": "error"`?
4. OOD питання отримують відмову, а не помилку Python?
5. `data/specialist_notes.md` заповнено?

---

## 🔥 Типові помилки

### `AttributeError: 'NoneType' has no attribute 'answer'`
Pydantic parse повернув None. Додай перевірку:
```python
if resp.choices[0].message.parsed is None:
    raise ValueError("Pydantic parse failed")
```

### Класифікатор завжди повертає "medium"
Перевір модель: `qwen2.5:1.5b`, а не `0.5b`. Якщо встановлена `0.5b`:
```bash
ollama pull qwen2.5:1.5b
```

### gpt-4o-mini відповідає занадто коротко
Додай в COT_SYSTEM: `"Розгорнута відповідь мінімум 3-5 речень."`

### Відповідь не на українській мові
Додай в системний промпт: `"ЗАВЖДИ відповідай виключно українською мовою."`

---

## ➡️ Наступний урок

[Урок 09: Побудова Eval Set](lesson_09_eval_set.md)
