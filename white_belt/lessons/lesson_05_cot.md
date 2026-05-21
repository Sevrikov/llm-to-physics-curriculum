# Урок 05: Chain-of-Thought
**Час:** ~8 годин | **Тиждень:** 2, День 11

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Знаєш коли CoT допомагає (і коли шкодить)
- ✅ Виміряв delta accuracy: direct vs CoT на 3 складних питаннях
- ✅ Спробував self-consistency — majority voting з N семплів
- ✅ Розумієш що CoT — інженерний прийом, а не "мислення"

---

## 📖 Теорія (15 хвилин)

### Що таке CoT насправді

⚠️ **Часта помилка:** "Модель думає покроково завдяки CoT."

Це невірно. Модель генерує токени послідовно — вона не "думає". CoT — це інженерний прийом: ти змушуєш модель **виробляти проміжний текст** перед фінальною відповіддю.

Механізм: кожен наступний токен "бачить" всі попередні через Attention. Довший проміжний контекст (розрахунки, кроки логіки) статистично покращує якість фінального токену.

**Аналогія:** Попроси когось порахувати 23×47 в умі — помиляться. Попроси записати кроки — якість зросте. Не тому що мозок "думає по-іншому", а тому що є де фіксувати проміжні стани.

### Коли CoT допомагає

| Тип задачі | CoT потрібен? | Чому |
|-----------|--------------|------|
| Математика, розрахунки | ✅ Так | Проміжні числа важливі |
| Логічні ланцюжки (3+ кроки) | ✅ Так | Кожен крок залежить від попереднього |
| Порівняння кількох сценаріїв | ✅ Так | Треба розглянути кожен |
| Простий факт ("скільки днів?") | ❌ Ні | Зайві токени = зайві гроші |
| Класифікація | ❌ Ні | CoT може ввести bias |
| Короткий QA | ❌ Ні | Уповільнює без користі |

---

## 💻 Практика: `05_cot_lab.py`

```python
# experiments/05_cot_lab.py
import ollama
import time

COMPLEX_QUESTIONS = [
    {
        "question": """Я працюю 3 роки, зарплата 25000 грн. Мене скорочують.
Яку суму компенсації я маю отримати і протягом якого терміну?""",
        "why_hard": "потрібен розрахунок: вихідна допомога + невикористана відпустка + строки"
    },
    {
        "question": """Роботодавець хоче перевести на дистанційку і зменшити зарплату на 20%.
Я проти. Які мої права і що може зробити роботодавець?""",
        "why_hard": "кілька сценаріїв і їх наслідки"
    },
    {
        "question": """Я на випробувальному 2 місяці. Роботодавець каже 'не підходиш'.
Що він може зробити? Які мої захисні дії?""",
        "why_hard": "права обох сторін + стратегія захисту"
    },
]

DIRECT_SYSTEM = "Ти юрист по трудовому праву України. Відповідай точно і конкретно."

COT_SYSTEM = """Ти юрист по трудовому праву України.
Для складних питань думай покроково ПЕРЕД відповіддю.

Формат:
<аналіз>
Крок 1: [що тут відбувається юридично]
Крок 2: [які норми застосовні]
Крок 3: [розрахунки якщо потрібні]
Крок 4: [можливі сценарії]
</аналіз>

**Відповідь:** [фінальна практична рекомендація]"""

for item in COMPLEX_QUESTIONS:
    print(f"\n{'='*70}")
    print(f"ПИТАННЯ: {item['question'][:80]}...")
    print(f"Складність: {item['why_hard']}")

    t0 = time.perf_counter()
    direct = ollama.chat(
        model='qwen2.5:7b',
        messages=[
            {'role': 'system', 'content': DIRECT_SYSTEM},
            {'role': 'user',   'content': item['question']},
        ],
        options={"temperature": 0},
    )
    direct_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    cot = ollama.chat(
        model='qwen2.5:7b',
        messages=[
            {'role': 'system', 'content': COT_SYSTEM},
            {'role': 'user',   'content': item['question']},
        ],
        options={"temperature": 0},
    )
    cot_time = time.perf_counter() - t0

    print(f"\n--- DIRECT ({direct_time:.1f}s, {len(direct['message']['content'])} chars) ---")
    print(direct['message']['content'][:400])

    print(f"\n--- CoT ({cot_time:.1f}s, {len(cot['message']['content'])} chars) ---")
    print(cot['message']['content'][:600])
```

---

## 🔁 Pro режим: Self-consistency (majority voting)

CoT з одного семпла — ненадійний для складних розрахунків. Модель може дати різний результат при незначній варіації формулювання. Self-consistency усуває цю проблему:

**Ідея:** запитати N разів з `temperature > 0`, зібрати N відповідей, взяти більшість.

```python
# experiments/05_self_consistency.py
import ollama
from collections import Counter
import re

COT_SYSTEM = """Ти юрист по трудовому праву України.
Думай покроково. Наприкінці обов'язково напиши:
ВІДПОВІДЬ: [одне речення — конкретний результат]"""

def extract_final_answer(text: str) -> str:
    """Виймає рядок після 'ВІДПОВІДЬ:'"""
    m = re.search(r'ВІДПОВІДЬ:\s*(.+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else text[-100:].strip()

def cot_with_consistency(question: str, n_samples: int = 3) -> dict:
    """
    Self-consistency: генеруємо N CoT traces, беремо більшість.
    temperature > 0 щоб отримати варіацію між семплами.
    """
    answers = []
    traces  = []

    for i in range(n_samples):
        resp = ollama.chat(
            model='qwen2.5:7b',
            messages=[
                {'role': 'system', 'content': COT_SYSTEM},
                {'role': 'user',   'content': question},
            ],
            options={"temperature": 0.5},  # НЕ 0 — потрібна варіативність
        )
        full_text = resp['message']['content']
        answer    = extract_final_answer(full_text)
        answers.append(answer)
        traces.append(full_text)
        print(f"  Семпл {i+1}: {answer[:80]}...")

    # Majority voting
    most_common_answer, votes = Counter(answers).most_common(1)[0]
    print(f"\n  Консенсус ({votes}/{n_samples} голосів): {most_common_answer}")

    return {
        "consensus": most_common_answer,
        "votes":     votes,
        "total":     n_samples,
        "all":       answers,
    }

# Тест на питанні де потрібен точний розрахунок
question = "Працюю 4 роки, зарплата 30000 грн/міс. Мене скорочують. Яка мінімальна сума вихідної допомоги?"
print(f"ПИТАННЯ: {question}\n")
result = cot_with_consistency(question, n_samples=3)
print(f"\nФінальна відповідь: {result['consensus']}")
print(f"Впевненість: {result['votes']}/{result['total']} семплів погодились")
```

> 💰 **Вартість:** self-consistency в 3× дорожче ніж звичайний CoT. Використовуй тільки для критичних розрахунків де потрібна точність.  
> **Коли НЕ використовувати:** прості факти, творчі завдання, класифікація.

---

## 🔍 Перевірка: чи висновок випливає з міркувань?

CoT може бути "красивою нісенітницею" — логічно звучить, але відповідь не витікає з міркувань. Перевір:

```python
# experiments/05_faithfulness_check.py
import ollama

def check_cot_faithfulness(reasoning: str, conclusion: str) -> bool:
    """Чи логічно випливає висновок з міркувань?"""
    judge_prompt = f"""Reasoning:
{reasoning}

Conclusion:
{conclusion}

Does the conclusion logically follow from the reasoning above?
Answer only: YES or NO"""

    resp = ollama.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': judge_prompt}],
        options={"temperature": 0},
    )
    return "yes" in resp['message']['content'].lower()

# Використання в eval loop:
# cot_response = ask_cot(question)
# reasoning  = extract_between_tags(cot_response, "<аналіз>", "</аналіз>")
# conclusion = extract_after(cot_response, "**Відповідь:**")
# if not check_cot_faithfulness(reasoning, conclusion):
#     print("⚠️ CoT unfaithful — висновок не випливає з міркувань!")
```

---

## 📝 Завдання (2 години)

```markdown
# data/cot_analysis.md

## Питання 1 (скорочення + розрахунок)
- Direct: [чи є конкретні суми?]
- CoT: [чи є конкретні суми і кроки?]
- Self-consistency (3 семпли): [чи погодились?]
- Переможець: ___

## Питання 2 (перевод + зарплата)
...

## Питання 3 (випробувальний термін)
...

## Загальний висновок
CoT потрібен у моїй ніші коли: ___
Self-consistency варта витрат коли: ___
CoT марний у моїй ніші коли: ___
```

---

## ✅ Самоперевірка

1. Можеш пояснити чому CoT ≠ "модель думає"?
2. На яких з 3 питань CoT дав кращий результат?
3. Запустив self-consistency — чи 3 семпли погодились між собою?
4. Знаєш для яких типів питань ніші будеш використовувати CoT vs direct?

---

## ⚠️ Типові помилки

| Помилка | Реальність |
|---------|-----------|
| "CoT = модель думає як людина" | CoT — generated text що корелює з кращими відповідями, не мислення |
| "CoT завжди покращує результат" | Для простих фактів — CoT додає шум, latency, cost без користі |
| "Довший CoT = кращий" | Concise CoT з правильною структурою > verbose рамблінг |
| "Self-consistency завжди вартує" | 3× дорожче. Тільки для критичних розрахунків де точність важлива |

---

## 🔥 Типові проблеми

**CoT на простих питаннях** — "Скільки днів відпустки?" не потребує `<аналіз>`. Це +30% токенів без користі.

**Занадто детальний `<аналіз>`** — модель "загубиться" в міркуваннях і дасть нерелевантну відповідь. Обмеж 3-4 кроки.

**extract_final_answer не знаходить відповідь** — модель не дотримується формату. Зроби промпт жорсткішим: `Останній рядок ОБОВ'ЯЗКОВО: ВІДПОВІДЬ: ...`

---

## ➡️ Наступний урок

[Урок 06: Structured Output](lesson_06_structured_output.md)

> CoT дає гарне міркування. Але що якщо треба не текст, а структуровані дані — JSON, таблицю, типізовані поля? Урок 06 — про надійний вихід.
