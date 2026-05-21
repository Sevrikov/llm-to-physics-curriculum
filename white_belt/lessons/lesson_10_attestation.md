# Урок 10: Фінальна атестація
**Час:** ~16 годин | **Тиждень:** 4, День 25–28

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Откалібрував LLM-суддю перед атестацією
- ✅ Запустив `run_eval.py` і отримав score >= 7.0/10
- ✅ Проаналізував помилки за таксономією і знаєш що виправляти
- ✅ Готовий до атестації Білого Поясу

---

## 📖 Теорія (10 хвилин)

### LLM-суддя: як це працює

```
питання + gold + твоя відповідь → GPT-4o-mini → оцінка 1-10
```

GPT-4o-mini порівнює відповідь `specialist.py` з еталоном і виставляє числову оцінку. Це дешевше і швидше ніж ручна перевірка — але тільки якщо суддя **калібрований**.

### Пороги атестації

```
< 5.0  — FAIL: суттєві проблеми в промпті або routing
5.0–6.9 — Прийнятно: є над чим працювати, але базова функціональність є
7.0–8.9 — ✅ PASS: Білий Пояс отримано
9.0+    — 🏆 З відзнакою
```

### Що оцінюється окремо

| Компонент | Вага | Мінімум |
|-----------|------|---------|
| Якість відповідей (LLM-judge avg) | 60% | ≥ 7.0 |
| OOD відмови коректні | 20% | ≥ 4/5 |
| Жодного Python exception | 10% | 100% |
| Швидкість: simple < 5s, medium < 15s | 10% | ≥ 80% |

---

## 🎯 Крок 0: Калібрування судді (перед атестацією)

GPT-4o-mini як суддя — не нейтральний. Він може систематично завищувати або занижувати оцінки. Перевір це перед основним eval:

```python
# experiments/10_calibrate_judge.py
"""
Калібруємо LLM-суддю: порівнюємо з людськими оцінками.
Кроки:
1. Оціни 10 відповідей ВРУЧНУ (5-10 хвилин)
2. Запусти suддю на тих самих відповідях
3. Порівняй кореляцію
"""
import json
from openai import OpenAI
from scipy.stats import pearsonr

client = OpenAI()

JUDGE_PROMPT = """Оціни відповідь асистента по шкалі 1-10.

Питання: {question}
Еталонна відповідь: {gold}
Відповідь асистента: {answer}

Критерії:
- 9-10: точна відповідь з нормами, корисна
- 7-8:  є неточності або пропуски
- 5-6:  частина key_facts відсутня
- 3-4:  суттєві помилки
- 1-2:  хибна або нерелевантна

Відповідь: тільки число 1-10."""

def judge_score(question: str, gold: str, answer: str) -> float:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ти суддя якості відповідей."},
            {"role": "user",   "content": JUDGE_PROMPT.format(
                question=question, gold=gold, answer=answer
            )},
        ],
        temperature=0,
    )
    try:
        return float(resp.choices[0].message.content.strip().split()[0])
    except Exception:
        return 5.0


# ── Заповни вручну перед запуском ──────────────────────────────
# 10 пар (відповідь, твоя_оцінка) — витрать 10 хвилин щоб оцінити чесно
CALIBRATION_SET = [
    {
        "question": "Скільки днів відпустки?",
        "gold":     "Мінімум 24 дні (ст. 75 КЗпП).",
        "answer":   "За законом належить 24 дні щорічної відпустки.",
        "human_score": 8,  # ← Заповни своє значення
    },
    {
        "question": "Мене звільнили незаконно. Що робити?",
        "gold":     "Ст. 235 КЗпП — суд, 1 місяць. Поновлення + компенсація.",
        "answer":   "Можна звернутись до суду.",
        "human_score": 4,  # ← Заповни своє значення
    },
    # ... додай ще 8 прикладів з data/eval_set.jsonl
]
# ────────────────────────────────────────────────────────────────

human_scores = [item["human_score"] for item in CALIBRATION_SET]
llm_scores   = [
    judge_score(item["question"], item["gold"], item["answer"])
    for item in CALIBRATION_SET
]

correlation, p_value = pearsonr(human_scores, llm_scores)

print(f"Judge calibration: r = {correlation:.2f} (p = {p_value:.4f})")
print(f"Human scores: {human_scores}")
print(f"LLM scores:   {[round(s, 1) for s in llm_scores]}")

if correlation < 0.7:
    print("\n⚠️ Суддя погано корелює з людиною (r < 0.7)")
    print("   Рішення: додай few-shot приклади оцінювання в JUDGE_PROMPT")
    print("   Наприклад: 'Ось приклад оцінки 8/10: [відповідь з нормою але без строків]'")
elif correlation >= 0.8:
    print("\n✅ Суддя добре калібрований (r >= 0.8) — можна довіряти оцінкам.")
else:
    print("\n💛 Прийнятна кореляція (0.7 <= r < 0.8) — результати орієнтовні.")
```

> **Мінімум для калібрації:** 10 питань, оцінених вручну. Це 10–15 хвилин. Якщо r < 0.7 — додай 2-3 few-shot приклади оцінювання в `JUDGE_PROMPT` і перевір знову.

---

## 💻 Практика: `run_eval.py`

```python
# run_eval.py — фінальна атестація specialist.py
import json
import time
from collections import Counter
from pathlib import Path
from openai import OpenAI
from specialist import ask

client = OpenAI()

JUDGE_PROMPT = """Оціни відповідь асистента по шкалі 1-10.

Питання: {question}
Еталонна відповідь: {gold_answer}
Ключові факти що мають бути: {key_facts}
Відповідь асистента: {system_answer}

Критерії оцінки:
- 9-10: всі key_facts присутні, відповідь точна, конкретна, корисна
- 7-8:  більшість key_facts є, є незначні неточності або пропуски
- 5-6:  деякі key_facts відсутні, або відповідь занадто загальна
- 3-4:  суттєві фактичні помилки або більшість key_facts відсутні
- 1-2:  відповідь хибна, нерелевантна або відмова там де треба відповідати

Для OOD питань (is_ood=true):
- 9-10: чітко відмовив і пояснив чому
- 1-2:  спробував відповісти по темі що не є в ніші

Відповідь: тільки одне число від 1 до 10."""


def judge_answer(question: str, gold: str, key_facts: list, answer: str) -> float:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ти суворий але справедливий суддя якості відповідей."},
                {"role": "user",   "content": JUDGE_PROMPT.format(
                    question=question, gold_answer=gold,
                    key_facts=key_facts, system_answer=answer,
                )},
            ],
            temperature=0,
        )
        return float(resp.choices[0].message.content.strip().split()[0])
    except Exception:
        return 5.0


def run_eval(eval_path: str = "data/eval_set.jsonl") -> dict:
    with open(eval_path, encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f]

    results = []
    errors  = 0

    print(f"Запуск eval: {len(eval_set)} прикладів\n{'='*65}")

    for i, item in enumerate(eval_set):
        q      = item["question"]
        gold   = item["gold_answer"]
        facts  = item.get("key_facts", [])
        is_ood = item.get("is_ood", False)

        t0 = time.perf_counter()
        try:
            result = ask(q)
            answer = result["answer"]
            if result["model_used"] == "error":
                errors += 1
        except Exception as e:
            answer = f"EXCEPTION: {e}"
            errors += 1
        elapsed = time.perf_counter() - t0

        # OOD перевірка
        ood_pass = None
        if is_ood:
            refusal_kw = ["поза", "не в моїй", "спеціалізац", "господарськ",
                          "рекоменд", "юрист", "бухгалтер", "не можу"]
            ood_pass = any(kw in answer.lower() for kw in refusal_kw)

        score = judge_answer(q, gold, facts, answer)

        results.append({
            "id":       item.get("id", f"q{i+1:03d}"),
            "question": q[:80],
            "score":    score,
            "elapsed":  elapsed,
            "is_ood":   is_ood,
            "ood_pass": ood_pass,
            "error":    result.get("model_used") == "error" if "result" in dir() else True,
            "answer":   answer[:200],
            "gold":     gold[:100],
            "category": item.get("category", ""),
        })

        icon     = "✅" if score >= 7.0 else "⚠️" if score >= 5.0 else "❌"
        ood_icon = f" [OOD: {'✅' if ood_pass else '❌'}]" if is_ood else ""
        print(f"[{i+1:02d}] {icon} {score:.0f}/10 {elapsed:4.1f}s{ood_icon} | {q[:55]}")

        time.sleep(0.2)

    # ── Підсумок ───────────────────────────────────────────────
    scores      = [r["score"] for r in results]
    avg_score   = sum(scores) / len(scores)
    pass_count  = sum(1 for s in scores if s >= 7.0)
    ood_results = [r for r in results if r["is_ood"]]
    ood_ok      = sum(1 for r in ood_results if r["ood_pass"])

    print(f"\n{'='*65}\nРЕЗУЛЬТАТИ АТЕСТАЦІЇ\n{'='*65}")
    print(f"Середній score:    {avg_score:.2f}/10")
    print(f"Питань >= 7.0:     {pass_count}/{len(results)} ({pass_count/len(results)*100:.0f}%)")
    print(f"OOD відмови:       {ood_ok}/{len(ood_results)}")
    print(f"Python errors:     {errors}/{len(results)}")

    # Розподіл по категоріях
    by_cat = {}
    for r in results:
        cat = r.get("category", "?")
        by_cat.setdefault(cat, []).append(r["score"])
    print("\nScore по категоріях:")
    for cat, cat_scores in sorted(by_cat.items()):
        print(f"  {cat:15s}: {sum(cat_scores)/len(cat_scores):.1f} (n={len(cat_scores)})")

    if avg_score >= 9.0:   verdict = "🏆 PASS З ВІДЗНАКОЮ"
    elif avg_score >= 7.0: verdict = "✅ PASS — Білий Пояс отримано!"
    elif avg_score >= 5.0: verdict = "⚠️  МАЙЖЕ — є над чим попрацювати"
    else:                  verdict = "❌ FAIL — потрібне суттєве доопрацювання"

    print(f"\nВЕРДИКТ: {verdict}\n{'='*65}")

    # Зберегти результати
    Path("data").mkdir(exist_ok=True)
    with open("data/eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Топ-5 найгірших для аналізу
    worst = sorted(results, key=lambda x: x["score"])[:5]
    print("\nТОП-5 НАЙГІРШИХ (для покращення):")
    for r in worst:
        print(f"\n  [{r['id']}][{r['category']}] Score: {r['score']}/10")
        print(f"  Питання: {r['question']}")
        print(f"  Еталон:  {r['gold']}")
        print(f"  Відповідь: {r['answer'][:150]}...")

    return {"avg_score": avg_score, "verdict": verdict}


if __name__ == "__main__":
    run_eval()
```

---

## 🔬 Аналіз помилок за таксономією

"Score 4" — незрозуміло чому. Класифікуй помилки щоб знати що виправляти:

```python
# experiments/10_error_taxonomy.py
import json
from openai import OpenAI

client = OpenAI()

ERROR_TYPES = {
    "hallucination":     "Факт неіснуючий або неправильний (неправильна стаття, сума)",
    "omission":          "Відповідь правильна але пропущено важливу частину (строки, норму)",
    "format_violation":  "Правильно але неправильний формат (немає структури, немає confidence)",
    "ood_failure":       "Питання поза нішею, але модель відповіла замість відмовити",
    "low_confidence":    "Відповідь без вказівки confidence де вона потрібна",
}

def classify_error(question: str, answer: str, gold: str) -> str:
    prompt = f"""Визнач тип помилки у відповіді асистента:

Питання: {question}
Еталон: {gold}
Відповідь асистента: {answer}

Типи помилок:
{chr(10).join(f"- {k}: {v}" for k, v in ERROR_TYPES.items())}

Відповідь: тільки назва типу (одне слово з переліку вище)."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    result = resp.choices[0].message.content.strip().lower()
    for key in ERROR_TYPES:
        if key in result:
            return key
    return "other"


def analyze_errors(eval_results_path: str = "data/eval_results.jsonl"):
    with open(eval_results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    low_score = [r for r in results if r["score"] < 7]
    if not low_score:
        print("✅ Немає відповідей з score < 7!")
        return

    error_counts = {k: 0 for k in ERROR_TYPES}
    error_counts["other"] = 0

    print(f"Аналізую {len(low_score)} відповідей з score < 7...\n")
    for r in low_score:
        etype = classify_error(r["question"], r["answer"], r["gold"])
        error_counts[etype] = error_counts.get(etype, 0) + 1

    print("Аналіз помилок:")
    print("="*50)
    for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            desc = ERROR_TYPES.get(etype, "інше")
            print(f"  {etype:20s}: {count:2d}  ({desc})")

    print(f"\nТопова причина низьких оцінок: {max(error_counts, key=error_counts.get)}")
    print("→ Виправляй саме її в наступній ітерації промпту")


if __name__ == "__main__":
    analyze_errors()
```

---

## 📝 Завдання

### Крок 1: Калібрування судді (1 год)
Заповни `CALIBRATION_SET` в `10_calibrate_judge.py` і запусти. Запиши r:
```
Judge calibration: r = ___ → ✅/⚠️/❌
```

### Крок 2: Перший eval запуск (2 год)
```bash
python run_eval.py
```
```
Перший score: ___ / 10
OOD pass: ___ / ___
```

### Крок 3: Аналіз помилок (2 год)
```bash
python experiments/10_error_taxonomy.py
```

```markdown
# data/attestation_analysis.md

## Перший score: ___ / 10
## Judge calibration r: ___

## Топ типи помилок:
1. ___ (кількість): рішення ___
2. ___ (кількість): рішення ___

## OOD аналіз: ___ / ___ пройшли
## Де провалилось: ___

## Зміни у specialist.py для 2-го запуску:
1. ___
2. ___
```

### Крок 4: Покращення та другий запуск (4 год)
```bash
python run_eval.py
```
```
Другий score: ___ / 10
Покращення: +___ балів
```

### Крок 5: Фінальна документація (2 год)

```markdown
# data/white_belt_completion.md

## Дата завершення: ___
## Фінальний score: ___ / 10
## Judge calibration r: ___
## Кількість ітерацій: ___

## Мій specialist.py — ключові рішення
### Ніша: ___
### Топ-3 елементи системного промпту: ___
### Few-shot: ___ прикладів, sweet spot ___-shot

## Що я навчився за Білий Пояс
1. ___
2. ___
3. ___

## Що планую в Жовтому Поясі
___
```

---

## 📌 Що далі після score >= 7.0

Score >= 7.0 — це не "запускай одразу в production". Це свідчить що система працює на eval set. Production deployment — інша справа:

```
Staged rollout (концепція для Yellow Belt):
  1. Тест на 5% реальних запитів → перевір що нічого не зламалось
  2. Моніторинг 48 годин → latency, error rate, user feedback
  3. Якщо ок → 50% → 100%

Для твого pet-project: тиждень тестування з 3-5 бета-юзерами.
Збирай реальні відповіді, порівнюй з eval.
```

---

## ✅ Самоперевірка

1. Judge calibration r >= 0.7?
2. `run_eval.py` запустився без помилок?
3. `data/eval_results.jsonl` містить 50 рядків?
4. Score >= 7.0 досягнуто?
5. OOD pass >= 4/5?
6. Запустив `10_error_taxonomy.py` і знаєш топову причину помилок?
7. `data/white_belt_completion.md` заповнено?

---

## ⚠️ Типові помилки

| Помилка | Реальність |
|---------|-----------|
| "LLM judge = об'єктивний" | LLM-судді мають biases: length bias, self-preference. Калібруй перед використанням |
| "Score 7.0 = добре скрізь" | Context-dependent. Медицина: 9.5+. Чатбот: 6.0 ок |
| "3 ітерації = convergence" | Може бути local optimum. Потрібна систематична exploration помилок |
| "Eval pass = deploy" | Staged rollout: 5% → 50% → 100% з моніторингом |

---

## 🔥 Якщо score < 7.0

### Score 5.0–6.9 — типові причини
| Проблема | Рішення |
|----------|---------|
| Відповіді занадто загальні | Додати в SYSTEM: "Завжди вказуй конкретну статтю закону" |
| Не вистачає key_facts | Додати few-shot приклад з максимальною конкретикою |
| OOD відмови слабкі | Додати явну OOD відмову в few-shot |
| Помилки типу "hallucination" | Додати: "Якщо не впевнений — скажи 'потребує уточнення'" |
| Помилки типу "format_violation" | Зроби шаблон відповіді жорсткішим в system prompt |

### Score < 5.0 — серйозніші проблеми
- Перечитай Урок 03 (системний промпт)
- Перечитай Урок 04 (few-shot)
- Запусти `02_break_the_prompt.py` знову

---

## 🎓 Вітаємо з Білим Поясом!

Ти пройшов повний цикл від "що таке промпт" до виміряної якості системи.

**Що ти вмієш тепер:**
- Будувати відтворювані baseline
- Захищати від prompt injection і jailbreak
- Проектувати системний промпт як конституцію поведінки
- Навчати модель на прикладах (включаючи негативні)
- Застосовувати CoT і self-consistency для складних задач
- Гарантувати надійний structured output
- Маршрутизувати між моделями для оптимізації cost/quality
- Будувати стратифікований eval set і калібрувати суддю
- Аналізувати помилки за таксономією

**Що далі:** [Belt 02: Жовтий Пояс](../../belt_02_yellow.md) — RAG, self-instruct pipeline, production observability, multi-agent базиси.
