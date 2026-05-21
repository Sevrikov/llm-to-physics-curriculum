# Урок 07: Cost-Quality Router *(Yellow Belt preview)*
**Час:** ~8 годин | **Тиждень:** 2, День 14

> 🎯 **Контекст:** Це preview теми Жовтого поясу. Тут зрозуміємо принцип. Повна production-версія — в belt_02_yellow.md, Техніка 6.

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Розумієш принцип routing: різні моделі для різної складності
- ✅ Маєш захист від неочікуваних рахунків (BudgetGuard)
- ✅ Маєш `07_cost_router.py` що показує реальну економію

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

## 🛡️ BudgetGuard: захист від неочікуваних рахунків

Один баг в коді — і замість $0.10 можна отримати $500 рахунок. BudgetGuard — простий денний ліміт:

```python
# experiments/budget_guard.py
import json, os
from datetime import date

class BudgetGuard:
    """Денний ліміт витрат — запобігає неочікуваним рахункам."""

    def __init__(self, daily_limit_usd: float = 5.0):
        self.limit      = daily_limit_usd
        self.state_file = ".budget_state.json"

    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                state = json.load(f)
            # Скидаємо стан якщо новий день
            if state.get("date") == str(date.today()):
                return state
        return {"date": str(date.today()), "spent": 0.0}

    def _save(self, state: dict):
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def record_spend(self, tokens_in: int, tokens_out: int,
                     price_in: float = 0.15, price_out: float = 0.60) -> bool:
        """Записує витрати. Повертає False якщо денний ліміт вичерпано."""
        state = self._load()
        cost  = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        state["spent"] += cost
        self._save(state)

        if state["spent"] > self.limit:
            print(f"⛔ Денний ліміт ${self.limit:.2f} вичерпано! "
                  f"Витрачено сьогодні: ${state['spent']:.4f}")
            return False
        return True

    def status(self) -> str:
        state = self._load()
        return f"Сьогодні: ${state['spent']:.4f} / ${self.limit:.2f}"
```

**Використання в router:**
```python
guard = BudgetGuard(daily_limit_usd=1.0)  # $1/день під час розробки

# Перед кожним дорогим запитом:
if not guard.record_spend(tokens_in, tokens_out, price_in=2.50, price_out=10.0):
    return ask_local(question)  # fallback на Ollama якщо ліміт вичерпано
```

---

## 💻 Практика: `07_cost_router.py`

```python
# experiments/07_cost_router.py
import ollama, tiktoken, time
from openai import OpenAI
from budget_guard import BudgetGuard

client = OpenAI()
enc    = tiktoken.encoding_for_model("gpt-4o-mini")
guard  = BudgetGuard(daily_limit_usd=2.0)

PRICING = {
    "local":       0.000,
    "gpt-4o-mini": 0.000375,  # avg (input+output) / 1M
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
        ],
        options={"temperature": 0},
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

total_routed    = 0.0
total_all_gpt4o = 0.0

for q in test_questions:
    complexity = classify(q)

    if complexity == 'simple':
        resp   = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}],
            options={"temperature": 0},
        )
        answer = resp['message']['content']
        model  = "local"
        # BudgetGuard: local = нема витрат, але все одно логуємо
        guard.record_spend(0, 0)

    elif complexity == 'medium':
        if not guard.record_spend(count_tokens(q), 200):
            # Ліміт! Fallback на local
            resp   = ollama.chat(model='qwen2.5:7b',
                messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}])
            answer = resp['message']['content']
            model  = "local_budget_fallback"
        else:
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                temperature=0,
            )
            answer = resp.choices[0].message.content
            model  = "gpt-4o-mini"

    else:  # complex
        if not guard.record_spend(count_tokens(q), 400, price_in=2.50, price_out=10.0):
            model  = "gpt-4o-mini"  # Деградуємо до mini замість повного fallback
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                temperature=0,
            )
            answer = resp.choices[0].message.content
        else:
            resp   = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                temperature=0,
            )
            answer = resp.choices[0].message.content
            model  = "gpt-4o"

    tokens     = count_tokens(q + answer)
    cost       = PRICING.get(model, PRICING["gpt-4o-mini"]) * tokens / 1000
    cost_gpt4o = PRICING["gpt-4o"] * tokens / 1000

    total_routed    += cost
    total_all_gpt4o += cost_gpt4o

    print(f"[{complexity:7s}] → {model:20s} | ${cost:.5f} | {q[:50]}...")

savings = (1 - total_routed / total_all_gpt4o) * 100 if total_all_gpt4o > 0 else 0
print(f"\nВитрати routed:    ${total_routed:.5f}")
print(f"Витрати all-GPT4o: ${total_all_gpt4o:.5f}")
print(f"Економія:          {savings:.0f}%")
print(f"\n{guard.status()}")
```

---

## 🔀 Cascade routing: три рівні замість двох

Бінарний вибір (local vs API) — грубо. Cascade дає плавну деградацію:

```python
def cascade_ask(question: str, guard: BudgetGuard) -> dict:
    """
    Спробуй дешеву модель спочатку.
    Ескалюй тільки якщо confidence низький або ліміт не вичерпано.
    """
    # Рівень 1: local Ollama (безкоштовно)
    local_resp = ollama.chat(
        model='qwen2.5:7b',
        messages=[
            {'role': 'system', 'content': SYSTEM + "\nНаприкінці: CONFIDENCE: high/medium/low"},
            {'role': 'user', 'content': question},
        ],
        options={"temperature": 0},
    )
    local_text = local_resp['message']['content']
    confidence = "high" if "confidence: high" in local_text.lower() else "low"

    if confidence == "high":
        return {"answer": local_text, "tier": "local", "cost_usd": 0}

    # Рівень 2: GPT-4o-mini (дешево)
    if not guard.record_spend(count_tokens(question), 200):
        return {"answer": local_text, "tier": "local_budget_fallback", "cost_usd": 0}

    mini_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0,
    )
    return {
        "answer":   mini_resp.choices[0].message.content,
        "tier":     "gpt-4o-mini",
        "cost_usd": 0.0005,
    }
    # Рівень 3 (GPT-4o) тут не потрібен для більшості White Belt use cases
    # Додаси в Yellow Belt де будуть задачі що дійсно вимагають GPT-4o
```

---

## 📝 Завдання

1. Запусти на своїх 20 питаннях з Уроку 01
2. Запиши: яка % питань йде на локальну модель?
3. Перевір вручну 3 "simple" відповіді — чи якість прийнятна?
4. Встанови для себе розумний `daily_limit_usd` на час розробки курсу

```markdown
# data/routing_analysis.md

## Розподіл по тирах (20 питань ніші):
- local (simple):     ___% 
- gpt-4o-mini (med):  ___%
- gpt-4o (complex):   ___%

## Якість simple відповідей (вибірка 3):
| Питання | Local відповідь прийнятна? |
|---------|--------------------------|
| ... | ✅/❌ |

## Економія vs all-GPT-4o: ___%
## Мій daily_limit під час розробки: $___
```

---

## ✅ Самоперевірка

1. Класифікатор використовує `qwen2.5:1.5b` (не 0.5b)?
2. Є `startswith` (не `in`) для перевірки?
3. BudgetGuard встановлений і тестований?
4. При вичерпанні ліміту — є graceful fallback (не Exception)?
5. Бачиш економію > 30% vs all-GPT-4o?

---

## ⚠️ Типові помилки

| Помилка | Реальність |
|---------|-----------|
| "Classifier = зайвий overhead" | Хороший classifier заощаджує 10x більше ніж коштує |
| "Завжди на найдешевшу модель" | User satisfaction важливий. Cheap but wrong = churn |
| "Routing — one-time setup" | Розподіл запитів змінюється. Класифікатор потребує оновлень |
| "Local = безкоштовно" | GPU, електрика, амортизація — реальні витрати, просто не per-request |

---

## ➡️ Наступний урок

[Урок 08: Збірка системи](lesson_08_assembly.md)

> Маєш всі компоненти окремо. Тепер — зібрати їх в один `specialist.py` що реально запускається.
