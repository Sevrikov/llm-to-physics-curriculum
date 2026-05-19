# Урок 07: Cost-Quality Router *(Yellow Belt preview)*
**Час:** ~8 годин | **Тиждень:** 2, День 14

> 🎯 **Контекст:** Це preview теми Жовтого поясу. Тут зрозуміємо принцип. Повна production-версія — в belt_02_yellow.md, Техніка 6.

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Розумієш принцип routing: різні моделі для різної складності
- ✅ Маєш `07_cost_router.py` що показує економію

---

## 📖 Теорія (10 хвилин)

### Проблема: GPT-4o коштує в 16x дорожче за GPT-4o-mini

```
GPT-4o:      $2.50 / 1M input tokens + $10.00 / 1M output
GPT-4o-mini: $0.15 / 1M input tokens + $0.60 / 1M output
Local Qwen:  $0.00
```

Більшість питань ("скільки днів відпустки?") не потребують GPT-4o. Routing = платити GPT-4o тільки за складні питання.

### Принцип

```
Питання → Класифікатор (дешева модель) → simple/medium/complex
    simple  → Qwen-7B (безкоштовно)
    medium  → GPT-4o-mini (~$0.001)
    complex → GPT-4o (~$0.015)
```

### ⚠️ Важливо для класифікатора

```python
# ❌ Погано — 0.5B занадто слабка + 'in' дає false positive:
model = 'qwen2.5:0.5b'
if 'simple' in answer: return 'simple'

# ✅ Добре — 1.5B + startswith:
model = 'qwen2.5:1.5b'
first_word = answer.strip().lower().split()[0]
if first_word.startswith('simple'): return 'simple'
```

---

## 💻 Практика: `07_cost_router.py`

```python
# experiments/07_cost_router.py
import ollama, tiktoken
from openai import OpenAI

client  = OpenAI()
enc     = tiktoken.encoding_for_model("gpt-4o-mini")

PRICING = {
    "local":       0.000,
    "gpt-4o-mini": 0.000375,  # (input+output) / 1M avg
    "gpt-4o":      0.006250,  # avg
}

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def classify(question: str) -> str:
    resp = ollama.chat(
        model='qwen2.5:1.5b',
        messages=[
            {'role': 'system', 'content': 'Reply ONE word: simple, medium, or complex.'},
            {'role': 'user',   'content': question},
        ]
    )
    w = resp['message']['content'].strip().lower().split()[0]
    if w.startswith('simple'):  return 'simple'
    if w.startswith('complex'): return 'complex'
    return 'medium'

SYSTEM = "Ти юридичний асистент по трудовому праву України."

test_questions = [
    "Скільки днів відпустки?",
    "Мене звільнили під час лікарняного. Що робити?",
    "Розрахуй компенсацію: 5 років, 32000 грн/міс, 18 невикористаних днів відпустки, скорочення",
]

total_routed = 0.0
total_all_gpt4o = 0.0

for q in test_questions:
    complexity = classify(q)
    
    if complexity == 'simple':
        resp   = ollama.chat(model='qwen2.5:7b', messages=[
            {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}
        ])
        answer = resp['message']['content']
        model  = "local"
    elif complexity == 'medium':
        resp   = client.chat.completions.create(model="gpt-4o-mini", messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": q}
        ], temperature=0.1)
        answer = resp.choices[0].message.content
        model  = "gpt-4o-mini"
    else:
        resp   = client.chat.completions.create(model="gpt-4o", messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": q}
        ], temperature=0.1)
        answer = resp.choices[0].message.content
        model  = "gpt-4o"
    
    tokens      = count_tokens(q + answer)
    cost        = PRICING[model] * tokens / 1000
    cost_gpt4o  = PRICING["gpt-4o"] * tokens / 1000
    
    total_routed   += cost
    total_all_gpt4o += cost_gpt4o
    
    print(f"[{complexity:7s}] → {model:12s} | ${cost:.5f} | {q[:50]}...")

savings = (1 - total_routed / total_all_gpt4o) * 100
print(f"\nВитрати routed:    ${total_routed:.5f}")
print(f"Витрати all-GPT4o: ${total_all_gpt4o:.5f}")
print(f"Економія:          {savings:.0f}%")
```

---

## 📝 Завдання

1. Запусти на своїх 20 питаннях з Уроку 01
2. Запиши: яка % питань йде на локальну модель?
3. Перевір вручну 3 "simple" відповіді — чи якість прийнятна?

---

## ✅ Самоперевірка

1. Класифікатор використовує `qwen2.5:1.5b` (не 0.5b)?
2. Є `startswith` (не `in`) для перевірки?
3. Бачиш економію > 30% vs all-GPT-4o?

---

## ➡️ Наступний урок

[Урок 08: Збірка системи](lesson_08_assembly.md)
