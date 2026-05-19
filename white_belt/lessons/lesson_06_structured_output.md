# Урок 06: Structured Output як контракт
**Час:** ~10 годин | **Тиждень:** 2, День 12–13

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Маєш `robust_json_caller.py` що **ніколи** не повертає `None`
- ✅ Розумієш різницю regex / JSONDecoder / Pydantic підходів
- ✅ Знаєш коли використовувати кожен

---

## 📖 Теорія (10 хвилин)

### Чому "просто попросити JSON" не працює

Моделі (особливо локальні) повертають:
```
Ось JSON що ти просив:
```json
{"answer": "..."}
```
Сподіваюсь це допомогло!
```

`json.loads()` впаде на цьому тексті. Потрібна витяжка JSON з довільного тексту.

### Три рівні надійності

```
Рівень 1: json.loads() напряму       → падає ~40% часу
Рівень 2: regex + JSONDecoder         → надійно для локальних моделей
Рівень 3: Pydantic response_format    → гарантія від OpenAI (cloud only)
```

**Правило:** для OpenAI API — завжди Pydantic. Для локальних — JSONDecoder fallback.

### ⚠️ Антипаттерн: `r'\{[^{}]*\}'`

```python
# ❌ ЗЛАМАЄТЬСЯ на nested JSON:
re.search(r'\{[^{}]*\}', '{"a": {"b": 1}}')
# → знайде тільки {"b": 1}, пропустить зовнішній об'єкт!

# ✅ Правильно — JSONDecoder.raw_decode():
decoder = json.JSONDecoder()
for i, ch in enumerate(text):
    if ch == '{':
        try:
            obj, _ = decoder.raw_decode(text, i)
            return obj  # знаходить ПЕРШИЙ валідний JSON будь-якої глибини
        except json.JSONDecodeError:
            continue
```

---

## 💻 Практика: два підходи

### Підхід A: Robust parser для локальних моделей

```python
# experiments/06_structured_output.py — скорочена версія
import json, re
from typing import Optional

def extract_json(text: str) -> Optional[dict]:
    # 1. Напряму
    try: return json.loads(text.strip())
    except: pass
    # 2. З ```json``` блоку
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    # 3. JSONDecoder.raw_decode (підтримує nested!)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except: continue
    return None
```

### Підхід B: Pydantic (OpenAI cloud)

```python
# experiments/06b_pydantic_structured.py
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

client = OpenAI()

class LegalAnalysis(BaseModel):
    category:   str        = Field(description="Категорія: wage_delay | termination | vacation | other")
    urgency:    str        = Field(description="Терміновість: high | medium | low")
    answer:     str        = Field(description="Відповідь 1-2 речення українською")
    actions:    List[str]  = Field(description="Конкретні кроки")
    confidence: str        = Field(description="Рівень: high | medium | low")

def analyze(question: str) -> LegalAnalysis:
    try:
        resp = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ти юридичний асистент по трудовому праву України."},
                {"role": "user",   "content": question},
            ],
            response_format=LegalAnalysis,
            temperature=0.1,
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        # Завжди повертаємо валідний об'єкт — ніколи None!
        return LegalAnalysis(
            category="other", urgency="medium",
            answer="Технічна помилка. Зверніться до юриста.",
            actions=["Зверніться до спеціаліста"],
            confidence="low",
        )

# Тест
result = analyze("Затримали зарплату місяць!")
print(result.model_dump_json(indent=2, ensure_ascii=False))
```

---

## 📝 Завдання (3 години)

1. Запусти обидва підходи на 5 питаннях своєї ніші
2. Перевір що `extract_json` не повертає `None` навіть якщо модель "балакає"
3. Визнач: яку схему (поля) JSON потрібна для твоєї ніші — і напиши свій `BaseModel`

```markdown
# data/structured_output_notes.md
## Моя Pydantic схема для ніші:
- field_1: ___ (тому що ___)
- field_2: ___
...

## Де structured output потрібен у моєму specialist.py:
___
```

---

## ✅ Самоперевірка

1. `extract_json('blah {"a": {"b": 1}} text')` повертає `{"a": {"b": 1}}` (не None)?
2. Pydantic модель визначена для своєї ніші?
3. Fallback завжди повертає валідний об'єкт?

---

## ➡️ Наступний урок

[Урок 07: Cost Router (preview)](lesson_07_cost_router.md)
