# Урок 06: Structured Output як контракт
**Час:** ~10 годин | **Тиждень:** 2, День 12–13

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Маєш `robust_json_caller.py` що **ніколи** не повертає `None`
- ✅ Знаєш 4 типи помилок structured output і різну стратегію для кожного
- ✅ Розумієш різницю regex / JSONDecoder / Pydantic підходів

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
# experiments/06_structured_output.py
import json, re
from typing import Optional

def extract_json(text: str) -> Optional[dict]:
    # 1. Напряму
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 2. З ```json``` блоку
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3. JSONDecoder.raw_decode (підтримує nested!)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except Exception:
                continue
    return None
```

### Підхід B: Pydantic + класифікація помилок (OpenAI cloud)

Проста `except Exception` — занадто груба. Різні типи помилок вимагають різної реакції:

```python
# experiments/06b_pydantic_structured.py
import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List

client = OpenAI()

class LegalAnalysis(BaseModel):
    category:   str       = Field(description="wage_delay | termination | vacation | other")
    urgency:    str       = Field(description="high | medium | low")
    answer:     str       = Field(description="Відповідь 1-2 речення українською")
    actions:    List[str] = Field(description="Конкретні кроки")
    confidence: str       = Field(description="high | medium | low")

def analyze(question: str) -> dict:
    try:
        resp = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ти юридичний асистент по трудовому праву України."},
                {"role": "user",   "content": question},
            ],
            response_format=LegalAnalysis,
            temperature=0,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            # beta.parse повертає None при refusal — це не виняток!
            return {"ok": False, "error_type": "content_filter",
                    "detail": "Модель відмовила відповідати"}
        return {"ok": True, "data": parsed.model_dump()}

    except ValidationError as e:
        # Тип 1: Schema mismatch — модель повернула неправильну структуру
        # Стратегія: retry з сильнішою інструкцією
        return {"ok": False, "error_type": "schema_mismatch", "detail": str(e)}

    except json.JSONDecodeError as e:
        # Тип 2: Broken JSON — рідко з cloud, частіше з local Ollama
        # Стратегія: retry з "return ONLY valid JSON, no commentary"
        return {"ok": False, "error_type": "invalid_json", "detail": str(e)}

    except Exception as e:
        err_str = str(e).lower()
        if "content_filter" in err_str or "policy" in err_str:
            # Тип 3: Content filter — модель відмовила через safety
            # Стратегія: rephrase question
            return {"ok": False, "error_type": "content_filter", "detail": str(e)}
        # Тип 4: API error — network, rate limit, etc.
        # Стратегія: exponential backoff
        return {"ok": False, "error_type": "api_error", "detail": str(e)}


def analyze_with_retry(question: str, max_retries: int = 2) -> dict:
    """Різна стратегія retry для кожного типу помилки."""
    result = analyze(question)
    if result["ok"]:
        return result

    error_type = result["error_type"]

    if error_type == "schema_mismatch" and max_retries > 0:
        # Додаємо сильнішу інструкцію і пробуємо ще раз
        stronger_q = question + "\n\nIMPORTANT: Respond ONLY with valid JSON matching the schema exactly."
        return analyze_with_retry(stronger_q, max_retries - 1)

    if error_type in ("api_error",) and max_retries > 0:
        import time
        time.sleep(2)
        return analyze_with_retry(question, max_retries - 1)

    # content_filter, invalid_json → не retry, повертаємо safe fallback
    return {
        "ok": False,
        "error_type": error_type,
        "data": LegalAnalysis(
            category="other", urgency="medium",
            answer="Не вдалось обробити запит. Зверніться до юриста.",
            actions=["Зверніться до спеціаліста"],
            confidence="low",
        ).model_dump(),
    }


# Тест
questions = [
    "Затримали зарплату місяць!",
    "Мене звільнили незаконно.",
    "Напиши мені рецепт пирога.",  # OOD — спровокуємо content_filter або fallback
]
for q in questions:
    result = analyze_with_retry(q)
    status = "✅" if result["ok"] else f"⚠️ {result.get('error_type', '?')}"
    print(f"{status} | {q[:40]}")
    if "data" in result:
        print(f"   → category={result['data']['category']}, confidence={result['data']['confidence']}")
```

---

## 📌 Реальний кейс: Writer + Instructor

> **Writer** (платформа корпоративного контенту) використовував LLM для обробки PDF-звітів і фінансових таблиць.  
> **Проблема:** прямий промптинг давав різну структуру щоразу → неможлива автоматизація пайплайну.  
> **Рішення:** Pydantic-схема → OpenAI Structured Output → self-repair loop через бібліотеку Instructor.  
> **Результат:** 100% відповідність вихідних даних схемі без ручної перевірки.

Instructor (від Jason Liu) — обгортка навколо OpenAI API що автоматично retry при ValidationError з feedback до моделі. Якщо в тебе >50 Pydantic викликів на день — варто поглянути: `pip install instructor`.

---

## 📝 Завдання (3 години)

1. Запусти обидва підходи на 5 питаннях своєї ніші
2. Навмисно спровокуй кожен тип помилки і перевір що стратегія спрацювала
3. Визнач яку схему JSON потрібна для твоєї ніші:

```markdown
# data/structured_output_notes.md

## Моя Pydantic схема для ніші:
- field_1: ___ (тому що ___)
- field_2: ___

## Таблиця помилок що бачив:

| Тип помилки | Коли трапилась | Стратегія |
|-------------|----------------|-----------|
| schema_mismatch | | retry + stronger instruction |
| invalid_json | | retry з "ONLY JSON" |
| content_filter | | rephrase або fallback |
| api_error | | sleep + retry |

## Де structured output потрібен у моєму specialist.py:
___
```

---

## ✅ Самоперевірка

1. `extract_json('blah {"a": {"b": 1}} text')` повертає `{"a": {"b": 1}}` (не None)?
2. Pydantic модель визначена для своєї ніші?
3. Fallback **завжди** повертає валідний об'єкт (ніколи None)?
4. Розрізняєш 4 типи помилок і знаєш різну стратегію для кожного?

---

## ⚠️ Типові помилки

| Помилка | Реальність |
|---------|-----------|
| "`json.loads` = надійно" | Nested JSON, unicode, markdown обгортки — ламається. JSONDecoder надійніший |
| "`beta.parse` = 100% valid" | При refusal повертає `None`, не помилку — перевіряй явно |
| "Structured output тільки для OpenAI" | Ollama + llama.cpp підтримують grammar-based constrained generation |
| "Схема = максимально детальна" | Over-specification = система що ламається на edge cases. Простіше = надійніше |

---

## ➡️ Наступний урок

[Урок 07: Cost Router (preview)](lesson_07_cost_router.md)

> Маєш надійний вивід. Тепер — як платити менше не жертвуючи якістю: routing між моделями.
