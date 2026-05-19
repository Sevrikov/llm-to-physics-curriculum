# Урок 10: Фінальна атестація
**Час:** ~16 годин | **Тиждень:** 4, День 25–28

---

## 🎯 Мета уроку

До кінця цього уроку ти:
- ✅ Запустив `run_eval.py` і отримав score >= 7.0/10
- ✅ Знаєш де `specialist.py` слабкий і маєш план покращення
- ✅ Готовий до атестації Білого Поясу

---

## 📖 Теорія (10 хвилин)

### LLM-суддя: як це працює

```
питання + gold + твоя відповідь → GPT-4o-mini → оцінка 1-10
```

GPT-4o-mini виступає "суддею": він порівнює відповідь `specialist.py` з еталоном і виставляє числову оцінку. Це дешевше і швидше ніж ручна перевірка, але потребує калібрування.

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

## 💻 Практика: `run_eval.py`

```python
# run_eval.py — фінальна атестація specialist.py
import json
import time
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# Імпортуємо з свого specialist.py
from specialist import ask

# ============================================================
# LLM-СУДДЯ
# ============================================================

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
    """LLM-суддя: повертає оцінку 1.0-10.0."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ти суворий але справедливий суддя якості відповідей."},
                {"role": "user", "content": JUDGE_PROMPT.format(
                    question=question,
                    gold_answer=gold,
                    key_facts=key_facts,
                    system_answer=answer,
                )},
            ],
            temperature=0.0,
        )
        score_text = resp.choices[0].message.content.strip()
        return float(score_text.split()[0])
    except Exception as e:
        print(f"  [judge error] {e}")
        return 5.0  # neutral fallback


# ============================================================
# ОСНОВНИЙ EVAL LOOP
# ============================================================

def run_eval(eval_path: str = "data/eval_set.jsonl") -> dict:
    with open(eval_path, encoding="utf-8") as f:
        eval_set = [json.loads(line) for line in f]

    results = []
    errors  = 0

    print(f"Запуск eval: {len(eval_set)} прикладів\n{'='*65}")

    for i, item in enumerate(eval_set):
        q       = item["question"]
        gold    = item["gold_answer"]
        facts   = item.get("key_facts", [])
        is_ood  = item.get("is_ood", False)

        t0 = time.time()
        try:
            result = ask(q)
            answer = result["answer"]
            if result["model_used"] == "error":
                errors += 1
        except Exception as e:
            answer = f"EXCEPTION: {e}"
            errors += 1

        elapsed = time.time() - t0

        # OOD перевірка: чи відмовив?
        ood_pass = None
        if is_ood:
            refusal_keywords = ["поза", "не в моїй", "спеціалізац", "господарськ",
                                "рекоменд", "юрист", "бухгалтер", "не можу"]
            ood_pass = any(kw in answer.lower() for kw in refusal_keywords)

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
        })

        icon = "✅" if score >= 7.0 else "⚠️" if score >= 5.0 else "❌"
        ood_icon = f" [OOD: {'✅' if ood_pass else '❌'}]" if is_ood else ""
        print(f"[{i+1:02d}] {icon} {score:.0f}/10 {elapsed:4.1f}s{ood_icon} | {q[:55]}")

        time.sleep(0.2)  # rate limit

    # ============================================================
    # ПІДСУМОК
    # ============================================================

    scores       = [r["score"] for r in results]
    avg_score    = sum(scores) / len(scores)
    pass_count   = sum(1 for s in scores if s >= 7.0)
    ood_results  = [r for r in results if r["is_ood"]]
    ood_pass_all = sum(1 for r in ood_results if r["ood_pass"])
    slow_count   = sum(1 for r in results
                       if not r["is_ood"] and r["elapsed"] > (5 if "simple" in r.get("question","") else 15))

    print(f"\n{'='*65}")
    print(f"РЕЗУЛЬТАТИ АТЕСТАЦІЇ")
    print(f"{'='*65}")
    print(f"Середній score:      {avg_score:.2f}/10")
    print(f"Питань >= 7.0:       {pass_count}/{len(results)} ({pass_count/len(results)*100:.0f}%)")
    print(f"OOD відмови:         {ood_pass_all}/{len(ood_results)}")
    print(f"Python errors:       {errors}/{len(results)}")
    print(f"{'='*65}")

    if avg_score >= 9.0:
        verdict = "🏆 PASS З ВІДЗНАКОЮ"
    elif avg_score >= 7.0:
        verdict = "✅ PASS — Білий Пояс отримано!"
    elif avg_score >= 5.0:
        verdict = "⚠️  МАЙЖЕ — є над чим попрацювати"
    else:
        verdict = "❌ FAIL — потрібне суттєве доопрацювання"

    print(f"ВЕРДИКТ: {verdict}")

    # Зберегти детальні результати
    Path("data").mkdir(exist_ok=True)
    with open("data/eval_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nДетальні результати → data/eval_results.jsonl")

    # Найгірші 5 для аналізу
    worst = sorted(results, key=lambda x: x["score"])[:5]
    print(f"\n{'='*65}")
    print("ТОП-5 НАЙГІРШИХ ВІДПОВІДЕЙ (для покращення):")
    for r in worst:
        print(f"\n  [{r['id']}] Score: {r['score']}/10")
        print(f"  Питання: {r['question']}")
        print(f"  Еталон:  {r['gold']}")
        print(f"  Відповідь: {r['answer'][:150]}...")

    return {
        "avg_score":   avg_score,
        "pass_count":  pass_count,
        "total":       len(results),
        "ood_pass":    ood_pass_all,
        "ood_total":   len(ood_results),
        "errors":      errors,
        "verdict":     verdict,
    }


if __name__ == "__main__":
    run_eval()
```

---

## 📝 Завдання

### Крок 1: Перший запуск (2 год)

```bash
python run_eval.py
```

Запиши перший результат:
```
Перший score: ___ / 10
OOD pass: ___ / 5
```

### Крок 2: Аналіз слабких місць (2 год)

Відкрий `data/eval_results.jsonl` і знайди паттерни низьких оцінок:

```markdown
# data/attestation_analysis.md

## Перший score: ___ / 10

## Паттерни низьких оцінок (< 5)
### Паттерн 1: ___
Приклади питань: ___
Причина: ___
Рішення: ___

### Паттерн 2: ___
...

## OOD аналіз
Пройшло: _/5
Де провалилось: ___
Виправлення у FEW_SHOT: ___

## Зміни у specialist.py
1. Зміна 1: ___
2. Зміна 2: ___
```

### Крок 3: Покращення та другий запуск (4 год)

Внеси зміни у `specialist.py` і запусти знову:

```bash
python run_eval.py
```

```
Другий score: ___ / 10
Покращення: +___ балів
```

### Крок 4: Фінальна здача (до досягнення >= 7.0)

Якщо score >= 7.0 → **атестація пройдена** 🎉

Якщо score < 7.0 → повторити Крок 2-3.

**Максимум 3 ітерації.** Якщо після 3 ітерацій score < 7.0 — зверніться до ментора або вивчіть відповідні уроки знову.

### Крок 5: Фінальна документація (2 год)

```markdown
# data/white_belt_completion.md

## Дата завершення: ___
## Фінальний score: ___ / 10
## Кількість ітерацій: ___

## Мій specialist.py — ключові рішення
### Ніша: ___
### Системний промпт — 3 найважливіших елементи:
1. ___
2. ___
3. ___

### Few-shot приклади:
1. ___
2. ___

### Де routing найчастіше помилявся:
___

## Що я навчився за Білий Пояс
1. ___
2. ___
3. ___

## Що планую в Жовтому Поясі
___
```

---

## ✅ Самоперевірка

1. `run_eval.py` запустився без помилок?
2. `data/eval_results.jsonl` містить 50 рядків?
3. Score >= 7.0 досягнуто?
4. OOD pass >= 4/5?
5. `data/white_belt_completion.md` заповнено?

---

## 🔥 Якщо score < 7.0

### Score 5.0-6.9 — типові причини
| Проблема | Рішення |
|----------|---------|
| Відповіді занадто загальні | Додати в SYSTEM: "Завжди вказуй конкретну статтю закону" |
| Не вистачає key_facts | Додати few-shot приклад з максимальною конкретикою |
| OOD відмови слабкі | Додати явну OOD відмову в few-shot |
| Complex → неправильний routing | Перевір класифікатор на проблемних питаннях |

### Score < 5.0 — серйозніші проблеми
- Перечитай Урок 03 (системний промпт)
- Перечитай Урок 04 (few-shot)
- Запусти `02_break_the_prompt.py` знову — можливо є нові слабкі місця

---

## 🎓 Вітаємо з Білим Поясом!

Ти пройшов повний цикл від "що таке промпт" до виміряної якості системи.

**Що далі:** [Belt 02: Жовтий Пояс](../../belt_02_yellow.md) — збір датасетів, self-instruct pipeline, LLM-judge з калібруванням.
