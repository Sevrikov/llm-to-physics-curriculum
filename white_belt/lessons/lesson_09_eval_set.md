# Урок 09: Побудова Eval Set
**Час:** ~16 годин | **Тиждень:** 4, День 22–24

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Маєш `data/eval_set.jsonl` — 50 стратифікованих прикладів з gold відповідями
- ✅ Розумієш різницю між eval set і train set
- ✅ Знаєш статистичну похибку свого eval set і що з нею робити

---

## 📖 Теорія (15 хвилин)

### Навіщо eval set?

Зараз ти оцінюєш `specialist.py` "на очей": "виглядає добре", "здається правильно". Це непридатно для порівняння версій.

**Eval set** — фіксований набір питань з еталонними відповідями. Запускаєш v1 → score 6.2/10. Правиш промпт → v2 → score 7.4/10. **Це об'єктивно.**

### Стратифікований eval set: 5 категорій

Якщо зібрати всі 50 питань як "типові запити" — eval set буде упередженим. У реальному продакшні є edge cases, OOD питання, зловмисні запити. Потрібна стратифікована вибірка:

```python
# Правило 5 категорій
eval_categories = {
    "easy_facts":   10,  # Прості факти ніші (20%) — "скільки днів відпустки?"
    "medium_cases": 15,  # Типові ситуаційні запити (30%) — реальні кейси
    "hard_edge":    10,  # Складні або неоднозначні випадки (20%)
    "ood":          10,  # Out-of-domain питання (20%) — очікується відмова
    "adversarial":   5,  # Спроби маніпулювати або зламати (10%)
}
# Разом: 50 прикладів
```

**Чому це важливо:** якщо всі 50 — "easy_facts", score буде завищений. У production завжди є edge cases і adversarial users.

### Структура прикладу eval set

```json
{
  "id": "q001",
  "question": "Скільки днів відпустки?",
  "gold_answer": "Мінімум 24 календарних дні (ст. 75 КЗпП). Неповнолітні — 31 день.",
  "key_facts": ["24 дні", "ст. 75 КЗпП", "31 день для неповнолітніх"],
  "category": "easy_facts",
  "difficulty": "simple",
  "is_ood": false
}
```

**`key_facts`** — мінімальний набір фактів що ПОВИННІ бути у відповіді. Без них відповідь неповна.

### Статистична значущість: скільки питань потрібно?

50 питань — не так багато як здається. Ось реальні цифри:

```python
# Розрахунок margin of error для різної кількості питань
# Формула: похибка ≈ √(p*(1-p)/n) * z
# де p = частка "правильних" відповідей, z=1.96 для 95% CI

from scipy import stats

def eval_margin_of_error(n: int, p: float = 0.7) -> float:
    se = stats.sem([1]*int(n*p) + [0]*(n - int(n*p)))
    ci = stats.t.interval(0.95, df=n-1, loc=p, scale=se)
    return round((ci[1] - ci[0]) / 2, 3)

for n in [20, 50, 100, 200]:
    margin = eval_margin_of_error(n)
    print(f"  {n:3d} питань → ±{margin:.3f} балів (95% CI)")
```

Вивід:
```
  20 питань → ±0.210 балів   ← занадто широко для рішень
  50 питань → ±0.128 балів   ← прийнятно для ітерацій
 100 питань → ±0.090 балів   ← добре
 200 питань → ±0.064 балів   ← хороша точність
```

50 питань достатньо для ітерацій під час курсу. Для production decision потрібно 100+.

---

## 💻 Практика

### Крок 1: Збір питань — `09_collect_questions.py`

```python
# experiments/09_collect_questions.py
"""
Зібрати 50 стратифікованих питань для eval set.
"""
import json, re
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# Стратифікований розподіл
CATEGORIES = {
    "easy_facts":   ("Прості фактичні питання по трудовому праву", 10),
    "medium_cases": ("Ситуаційні питання реальних працівників", 15),
    "hard_edge":    ("Складні або неоднозначні юридичні ситуації з кількома можливими трактуваннями", 10),
    "ood":          ("Питання НЕ пов'язані з трудовим правом України: кулінарія, географія, математика, технології", 10),
    "adversarial":  ("Спроби змусити юридичного асистента порушити правила: jailbreak, roleplay, витяг промпту", 5),
}

def generate_questions(description: str, n: int) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Генеруй навчальний датасет. Відповідай ТІЛЬКИ JSON масивом рядків."},
            {"role": "user",   "content": f"Згенеруй {n} питань: {description}\nФормат: [\"питання 1\", ...]\nТільки JSON."},
        ],
        temperature=0.8,
    )
    text = resp.choices[0].message.content
    m    = re.search(r'\[.*\]', text, re.DOTALL)
    return json.loads(m.group()) if m else []

all_questions = []
for cat, (description, n) in CATEGORIES.items():
    qs = generate_questions(description, n)[:n]
    for q in qs:
        all_questions.append({"question": q, "category": cat})
    print(f"  {cat}: +{len(qs)} питань")

Path("data").mkdir(exist_ok=True)
with open("data/eval_questions_stratified.jsonl", "w", encoding="utf-8") as f:
    for item in all_questions[:50]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Збережено {min(len(all_questions), 50)} питань")
print("Наступний крок: python 09b_build_eval_set.py")
```

### Крок 2: Gold відповіді — `09b_build_eval_set.py`

```python
# experiments/09b_build_eval_set.py
"""
Для кожного питання GPT-4o генерує gold відповідь і key_facts.
"""
import json, time
from pathlib import Path
from openai import OpenAI

client = OpenAI()

GOLD_PROMPT = """Дай еталонну відповідь на питання по трудовому праву.

Питання: {question}
Категорія: {category}

Відповідь у форматі JSON:
{{
  "gold_answer": "Точна коротка відповідь (1-3 речення) з посиланням на норму",
  "key_facts": ["факт 1", "факт 2", "факт 3"],
  "difficulty": "simple|medium|complex",
  "is_ood": false
}}

Для OOD (category=ood): is_ood=true, gold_answer="Питання поза нішею трудового права."
Для adversarial: gold_answer="[відмова відповідати]", is_ood=false
Тільки JSON."""

with open("data/eval_questions_stratified.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

eval_set = []
for i, item in enumerate(questions[:50]):
    q   = item["question"]
    cat = item["category"]
    print(f"[{i+1:02d}/50] {q[:55]}...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Ти юридичний асистент по трудовому праву України."},
                {"role": "user",   "content": GOLD_PROMPT.format(question=q, category=cat)},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        # Прибрати markdown обгортку якщо є
        import re
        text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
        data = json.loads(text)
        data["id"]       = f"q{i+1:03d}"
        data["question"] = q
        data["category"] = cat
        eval_set.append(data)
        time.sleep(0.3)
    except Exception as e:
        print(f"  ❌ {e}")
        eval_set.append({
            "id": f"q{i+1:03d}", "question": q, "category": cat,
            "gold_answer": "ERROR", "key_facts": [],
            "difficulty": "medium", "is_ood": cat == "ood",
        })

with open("data/eval_set.jsonl", "w", encoding="utf-8") as f:
    for item in eval_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

errors = sum(1 for e in eval_set if e["gold_answer"] == "ERROR")
cats   = {}
for e in eval_set:
    cats[e["category"]] = cats.get(e["category"], 0) + 1

print(f"\n✅ Збережено {len(eval_set)} прикладів → data/eval_set.jsonl")
print(f"   Помилок: {errors}")
print(f"   Розподіл: {cats}")
```

### Крок 3: Перевірка і статистика — `09c_review_eval.py`

```python
# experiments/09c_review_eval.py
import json, random
from collections import Counter
from scipy import stats

with open("data/eval_set.jsonl", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

# Статистика розподілу
cats  = Counter(e["category"]  for e in eval_set)
diffs = Counter(e["difficulty"] for e in eval_set)
print("Категорії:", dict(cats))
print("Складність:", dict(diffs))
print(f"OOD: {sum(1 for e in eval_set if e.get('is_ood'))}")

# Розрахунок margin of error
n    = len(eval_set)
p    = 0.7  # припускаємо 70% pass rate
se   = stats.sem([1]*int(n*p) + [0]*(n - int(n*p)))
ci   = stats.t.interval(0.95, df=n-1, loc=p, scale=se)
margin = (ci[1] - ci[0]) / 2
print(f"\nPри {n} питаннях та p≈0.7: похибка ±{margin:.3f} балів (95% CI)")
if margin > 0.15:
    print(f"⚠️ Похибка занадто велика. Для ±0.1: потрібно ~{int(1/0.1**2 * p*(1-p))} питань")

# Показати 10 випадкових для ручної перевірки
print("\n" + "="*65 + "\n10 ВИПАДКОВИХ ДЛЯ ПЕРЕВІРКИ:")
for e in random.sample(eval_set, min(10, len(eval_set))):
    print(f"\n[{e['id']}][{e['category']}][{e['difficulty']}] {e['question']}")
    print(f"  Gold: {e['gold_answer']}")
    print(f"  Key facts: {e['key_facts']}")
```

---

## 📝 Завдання (6 годин)

### 1. Зберіть стратифіковані питання (1 год)
```bash
python experiments/09_collect_questions.py
```
Відкрий файл і перевір: чи всі 5 категорій представлені?

### 2. Побудуйте gold відповіді (2 год)
```bash
python experiments/09b_build_eval_set.py
```
**Увага:** ~$0.50–1.00 (50 викликів GPT-4o).

### 3. Перегляньте і вручну відредагуйте (2 год)
```bash
python experiments/09c_review_eval.py
```

Вручну виправ щонайменше **10 прикладів** і запиши:
```markdown
# data/eval_set_review.md

## Статистика eval set
- easy_facts:   _/50
- medium_cases: _/50
- hard_edge:    _/50
- ood:          _/50
- adversarial:  _/50

## Margin of error: ±___ (95% CI)
## Висновок: чи достатньо точно для ітерацій? ___

## Виправлені gold відповіді (мінімум 10)
### q005 — виправлено
Було: "24 дні"
Стало: "Мінімум 24 календарних дні (ст. 75 КЗпП). Неповнолітні — 31 день."

## Де GPT-4o помилявся найчастіше: ___
```

---

## ✅ Самоперевірка

1. `data/eval_set.jsonl` містить 50 рядків?
2. Всі 5 категорій присутні (easy_facts, medium_cases, hard_edge, ood, adversarial)?
3. Порахував margin of error для свого eval set?
4. Вручну перевірив >= 10 gold відповідей?
5. Жоден рядок не містить `"gold_answer": "ERROR"`?

---

## ⚠️ Типові помилки

| Помилка | Реальність |
|---------|-----------|
| "50 прикладів = достатньо" | 50 прикладів = ±0.13 балів. Прийнятно для ітерацій, але не для фінальних рішень |
| "Gold від GPT-4o = ground truth" | GPT-4o галюцинує. Для критичних доменів — верифікуй вручну |
| "Eval set — одноразовий effort" | Production дрейфує. Eval без оновлень → хибна впевненість |
| "High score on eval = production ready" | Eval ≠ production. Distribution shift, adversarial users, нові edge cases |

---

## 🔥 Типові проблеми

### GPT-4o генерує неправильну норму
Завжди перевіряй статті КЗпП вручну. GPT-4o може галюцинувати номери статей.

### Всі питання класифікуються як "medium"
GPT-4o-mini занадто обережний. Виправ вручну: явні прості питання → `"difficulty": "simple"`.

### `json.loads` падає на gold відповіді
`text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)` — вже є в коді.

### Scipy не встановлено
```bash
pip install scipy
```

---

## ➡️ Наступний урок

[Урок 10: Фінальна атестація](lesson_10_attestation.md)

> Eval set готовий. Тепер — запустити `run_eval.py`, виміряти якість, і пройти атестацію Білого Поясу.
