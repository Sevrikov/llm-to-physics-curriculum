# Урок 09: Побудова Eval Set
**Час:** ~16 годин | **Тиждень:** 4, День 22–24

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Маєш `data/eval_set.jsonl` — 50 прикладів з gold відповідями
- ✅ Розумієш різницю між eval set і train set
- ✅ Знаєш як оцінювати якість відповідей числово

---

## 📖 Теорія (15 хвилин)

### Навіщо eval set?

Зараз ти оцінюєш `specialist.py` "на очей": "виглядає добре", "здається правильно". Це непридатно для порівняння версій.

**Eval set** — фіксований набір питань з еталонними відповідями. Запускаєш v1 → score 6.2/10. Правиш промпт → v2 → score 7.4/10. **Це об'єктивно.**

### Структура eval set

```json
{
  "id": "q001",
  "question": "Скільки днів відпустки?",
  "gold_answer": "Мінімум 24 календарних дні (ст. 75 КЗпП). Неповнолітні — 31 день.",
  "key_facts": ["24 дні", "ст. 75 КЗпП", "31 день для неповнолітніх"],
  "category": "vacation",
  "difficulty": "simple"
}
```

**`key_facts`** — мінімальний набір фактів що ПОВИННІ бути у відповіді. Без них відповідь неповна.

### 50 прикладів: як набрати

```
10 × simple   — прості фактичні питання
20 × medium   — ситуативні питання
15 × complex  — розрахунки та складні сценарії
5  × OOD      — питання поза нішею (очікується відмова)
```

### Оцінка через LLM-суддя

```python
JUDGE_PROMPT = """Оціни відповідь по шкалі 1-10.

Питання: {question}
Еталонна відповідь: {gold_answer}
Ключові факти що мають бути: {key_facts}

Відповідь системи: {system_answer}

Критерії:
- 9-10: всі key_facts є, відповідь точна і корисна
- 7-8: більшість key_facts є, невеликі неточності
- 5-6: частина key_facts відсутня
- 3-4: суттєві помилки або ключові факти відсутні
- 1-2: відповідь хибна або нерелевантна

Відповідь: тільки число від 1 до 10."""
```

---

## 💻 Практика

### Крок 1: Збір питань — `09_collect_questions.py`

```python
# experiments/09_collect_questions.py
"""
Крок 1: Зібрати 50 питань для eval set.
Питання беруться з:
1. data/questions.txt (твої 20 з Уроку 01)
2. Генерація додаткових через GPT-4o-mini
"""
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# --- Зчитати свої 20 питань ---
with open("data/questions.txt", encoding="utf-8") as f:
    existing = [line.strip() for line in f if line.strip()]

print(f"Є {len(existing)} питань. Генеруємо додаткові...")

# --- Генерація додаткових питань ---
def generate_questions(category: str, n: int) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ти генеруєш навчальний датасет. Відповідай тільки JSON масивом рядків."},
            {"role": "user", "content": f"""Згенеруй {n} реалістичних питань по трудовому праву України.
Тип: {category}
Формат: ["питання 1", "питання 2", ...]
Тільки JSON, без пояснень."""}
        ],
        temperature=0.8,
    )
    import re, json as j
    text = resp.choices[0].message.content
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        return j.loads(m.group())
    return []

categories = {
    "simple_vacation": 5,   # прості питання про відпустку
    "simple_salary": 5,     # прості питання про зарплату
    "medium_dismissal": 10, # ситуативні питання про звільнення
    "complex_calc": 5,      # розрахункові задачі
    "ood": 5,               # питання поза нішею
}

all_questions = list(existing[:20])  # беремо перші 20 існуючих

for cat, n in categories.items():
    new_qs = generate_questions(cat, n)
    all_questions.extend(new_qs[:n])
    print(f"  {cat}: +{len(new_qs[:n])} питань")

# Зберегти
Path("data").mkdir(exist_ok=True)
with open("data/eval_questions.txt", "w", encoding="utf-8") as f:
    for q in all_questions[:50]:
        f.write(q + "\n")

print(f"\n✅ Збережено {min(len(all_questions), 50)} питань → data/eval_questions.txt")
print("Наступний крок: python 09b_build_eval_set.py")
```

### Крок 2: Побудова gold відповідей — `09b_build_eval_set.py`

```python
# experiments/09b_build_eval_set.py
"""
Крок 2: Для кожного питання GPT-4o генерує gold відповідь і key_facts.
"""
import json, time
from pathlib import Path
from openai import OpenAI

client = OpenAI()

SYSTEM = "Ти юридичний асистент по трудовому праву України. Відповідай точно і конкретно."

GOLD_PROMPT = """Дай еталонну відповідь на питання по трудовому праву.

Питання: {question}

Відповідь у форматі JSON:
{{
  "gold_answer": "Точна коротка відповідь (1-3 речення) з посиланням на норму",
  "key_facts": ["факт 1", "факт 2", "факт 3"],
  "category": "vacation|wage|dismissal|sick_leave|other|ood",
  "difficulty": "simple|medium|complex",
  "is_ood": false
}}

Для OOD питань: is_ood=true, gold_answer="Питання поза нішею трудового права."
Тільки JSON, без коментарів."""

with open("data/eval_questions.txt", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

eval_set = []
for i, q in enumerate(questions[:50]):
    print(f"[{i+1:02d}/50] {q[:60]}...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",       # gpt-4o для якісних gold відповідей
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": GOLD_PROMPT.format(question=q)},
            ],
            temperature=0.1,
        )
        import json as j
        data = j.loads(resp.choices[0].message.content.strip())
        data["id"]       = f"q{i+1:03d}"
        data["question"] = q
        eval_set.append(data)
        time.sleep(0.3)  # rate limit
    except Exception as e:
        print(f"  ❌ Помилка: {e}")
        eval_set.append({
            "id": f"q{i+1:03d}", "question": q,
            "gold_answer": "ERROR", "key_facts": [],
            "category": "other", "difficulty": "medium", "is_ood": False
        })

with open("data/eval_set.jsonl", "w", encoding="utf-8") as f:
    for item in eval_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

errors = sum(1 for e in eval_set if e["gold_answer"] == "ERROR")
print(f"\n✅ Збережено {len(eval_set)} прикладів → data/eval_set.jsonl")
print(f"   Помилок: {errors}")
print(f"   OOD: {sum(1 for e in eval_set if e.get('is_ood'))}")
```

### Крок 3: Перевірка eval set — `09c_review_eval.py`

```python
# experiments/09c_review_eval.py
"""
Крок 3: Переглянути 10 випадкових прикладів.
Ручна перевірка: чи gold відповіді правильні?
"""
import json, random

with open("data/eval_set.jsonl", encoding="utf-8") as f:
    eval_set = [json.loads(line) for line in f]

# Показати статистику
from collections import Counter
cats  = Counter(e["category"]  for e in eval_set)
diffs = Counter(e["difficulty"] for e in eval_set)
print("Категорії:", dict(cats))
print("Складність:", dict(diffs))
print(f"OOD: {sum(1 for e in eval_set if e.get('is_ood'))}/50")

# Показати 10 випадкових для ручної перевірки
print("\n" + "="*65)
print("10 ВИПАДКОВИХ ДЛЯ ПЕРЕВІРКИ:")
print("="*65)
for e in random.sample(eval_set, min(10, len(eval_set))):
    print(f"\n[{e['id']}] [{e['difficulty']}] {e['question']}")
    print(f"Gold: {e['gold_answer']}")
    print(f"Key facts: {e['key_facts']}")
```

---

## 📝 Завдання (6 годин)

### 1. Зберіть 50 питань (1 год)
```bash
python experiments/09_collect_questions.py
```
Відкрий `data/eval_questions.txt` і вручну перевір: чи питання реалістичні?

### 2. Побудуйте gold відповіді (2 год)
```bash
python experiments/09b_build_eval_set.py
```
**Увага:** Це коштує ~$0.50-1.00 (50 викликів GPT-4o).

### 3. Перегляньте і відредагуйте (2 год)
```bash
python experiments/09c_review_eval.py
```

Відкрий `data/eval_set.jsonl` і вручну виправ щонайменше **10 прикладів**:
```markdown
# data/eval_set_review.md

## Виправлені gold відповіді
### q005 — виправлено
Було: "24 дні"
Стало: "Мінімум 24 календарних дні (ст. 75 КЗпП). Для неповнолітніх — 31 день."
Причина: забракло конкретики

### q017 — виправлено
...

## Загальний висновок
- Де GPT-4o помилявся найчастіше: ___
- Яких key_facts не вистачало: ___
```

### 4. Збережіть фінальний eval set (1 год)

Після ручного редагування:
```
data/eval_set.jsonl — фінальна версія, 50 прикладів
data/eval_set_review.md — нотатки по якості
```

---

## ✅ Самоперевірка

1. `data/eval_set.jsonl` містить 50 рядків?
2. Є питання всіх типів: simple / medium / complex / OOD?
3. Вручну перевірив >= 10 gold відповідей?
4. `data/eval_set_review.md` заповнено?
5. Жоден рядок в jsonl не містить `"gold_answer": "ERROR"`?

---

## 🔥 Типові помилки

### GPT-4o генерує неправильну норму
Завжди перевіряй статті КЗпП вручну для своєї ніші. GPT-4o може галюцинувати номери статей.

### Всі питання класифікуються як "medium"
GPT-4o-mini занадто обережний. Виправ вручну в jsonl: явні прості питання ("скільки днів?") → `"difficulty": "simple"`.

### `json.loads` падає на gold відповіді
GPT-4o іноді повертає markdown обгортку. Або:
```python
# Чистити обгортку перед парсингом:
text = resp.choices[0].message.content.strip()
text = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.MULTILINE)
```

---

## ➡️ Наступний урок

[Урок 10: Фінальна атестація](lesson_10_attestation.md)
