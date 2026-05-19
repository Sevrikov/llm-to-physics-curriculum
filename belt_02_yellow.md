# 🟡 ЖОВТИЙ ПОЯС — «Майстер даних»
## Повний навчальний модуль

**Тривалість:** 6 тижнів / 240 годин  
**Передумови:** Білий пояс отримано (LLM-judge score >= 7.0/10), базовий Python  
**Що на виході:** Датасет 2000+ прикладів, якість >= 8.0/10, production-ready Cost-Quality Router  
**Бюджет:** ~$15–50 на API виклики (self-instruct + eval)

---

## КЛЮЧОВА ІДЕЯ ЖОВТОГО ПОЯСУ

> **«Якість даних визначає стелю моделі. Найкращий fine-tuning на поганих даних дасть погану модель.»**

На Білому поясі ти навчився давати команди моделі (промпти). Тут ти навчишся **годувати модель** — будувати датасети якість яких можна виміряти і захистити.

Суперник для атестації: **власна модель, натренована на поганому датасеті**. Ти маєш зібрати кращий датасет і довести що твоя версія краща. Не проти GPT-4o — проти себе.

---

## ПЕРЕД ПОЧАТКОМ: Продовж нішу з Білого поясу

Якщо у тебе вже є `data/first_fight_results.jsonl` і `evaluation/eval_gold.jsonl` — відмінно, будуємо на їх основі. Якщо ні — витрать 2 години і зроби Тиждень 1 Білого поясу.

```bash
# Перевірка готовності
python -c "
import json
from pathlib import Path

baseline = Path('data/first_fight_results.jsonl')
eval_set = Path('evaluation/eval_gold.jsonl')

print(f'Baseline: {sum(1 for _ in open(baseline))} питань' if baseline.exists() else '❌ Немає baseline')
print(f'Eval set: {sum(1 for _ in open(eval_set))} прикладів' if eval_set.exists() else '❌ Немає eval set')
"
```

---

## ТИЖДЕНЬ 1: ПЕРШИЙ БІЙ — «Два датасети, одна модель»

### День 1–3: Знайти, завантажити, порівняти (24 години)

**Мета:** Усвідомити що дані важливіші за архітектуру — один і той самий fine-tuning на різних датасетах дає радикально різний результат.

```python
# experiments/08_dataset_battle.py
"""
ПЕРШИЙ БІЙ РІВНЯ: завантажити 3 публічних датасети,
зробити мінімальний fine-tune на кожному, порівняти якість.

УВАГА: Повний fine-tune займає години і потребує GPU.
Тут ми СИМУЛЮЄМО ефект через few-shot demonstration —
щоб побачити принцип ДО того як матимемо GPU.
"""
import json
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI

client = OpenAI()

# ══════════════════════════════════════════════════════════════
# Знайти 3 датасети на HuggingFace що близькі до твоєї ніші
# Для юридичної ніші — приклади нижче. Замни на свої.
# Шукати: https://huggingface.co/datasets?search=legal+qa
# ══════════════════════════════════════════════════════════════

DATASETS_TO_COMPARE = {
    "formal_qa": {
        "description": "Формальні юридичні QA (офіційний стиль)",
        "examples": [
            {"q": "Що таке трудовий договір?",
             "a": "Трудовий договір — це угода між працівником та роботодавцем, відповідно до якої працівник зобов'язується виконувати роботу, а роботодавець — виплачувати заробітну плату."},
            {"q": "Скільки днів відпустки передбачено законом?",
             "a": "Відповідно до ст. 75 КЗпП України, мінімальна щорічна оплачувана відпустка становить 24 календарних дні."},
        ]
    },
    "casual_qa": {
        "description": "Спрощений стиль (як для WhatsApp-чату)",
        "examples": [
            {"q": "Що таке трудовий договір?",
             "a": "Це папір де ти і начальник домовляєтесь про роботу і зарплату. Без нього краще не виходити на роботу."},
            {"q": "Скільки днів відпустки?",
             "a": "24 дні мінімум по закону. Але в деяких компаніях дають більше — перевір свій договір."},
        ]
    },
    "no_domain": {
        "description": "Загальний QA без домену (навмисно поганий)",
        "examples": [
            {"q": "Що таке трудовий договір?",
             "a": "Це контракт. Він важливий. Потрібно його підписати."},
            {"q": "Скільки днів відпустки?",
             "a": "Залежить від компанії. Зазвичай кілька тижнів."},
        ]
    }
}

TEST_QUESTIONS = [
    "Роботодавець затримав зарплату. Що робити?",
    "Мене скорочують. На яку компенсацію маю право?",
    "Чи можна звільнити вагітну?",
]

print("🥊 БІЙ ДАТАСЕТІВ\n")
results = {}

for dataset_name, dataset_info in DATASETS_TO_COMPARE.items():
    print(f"\n{'='*60}")
    print(f"Датасет: {dataset_name} — {dataset_info['description']}")
    results[dataset_name] = []
    
    for question in TEST_QUESTIONS:
        # Симулюємо fine-tuning через few-shot з прикладами датасету
        few_shot_msgs = []
        for ex in dataset_info["examples"]:
            few_shot_msgs.append({"role": "user",    "content": ex["q"]})
            few_shot_msgs.append({"role": "assistant","content": ex["a"]})
        
        messages = [
            {"role": "system", "content": "Ти юридичний асистент по трудовому праву України."},
            *few_shot_msgs,
            {"role": "user", "content": question}
        ]
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.1
        )
        answer = resp.choices[0].message.content
        results[dataset_name].append({"question": question, "answer": answer})
        print(f"\nQ: {question[:50]}...")
        print(f"A: {answer[:200]}...")

# Зберегти для порівняння
Path("data").mkdir(exist_ok=True)
with open("data/dataset_battle_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n\n✅ Збережено: data/dataset_battle_results.json")
print("\nЗАВДАННЯ (2 години):")
print("Відкрий файл і для кожного датасету оціни відповіді:")
print("1. Яка відповідь найкорисніша для реального користувача?")
print("2. Який стиль відповідає твоїй ніші?")
print("3. Що є в поганому датасеті що робить відповіді гіршими?")
```

#### Аналіз результатів (2 години)

Заповни після запуску:

```markdown
# data/dataset_comparison_report.md

## Висновки з порівняння датасетів

### Що робить датасет ХОРОШИМ для моєї ніші:
1. [конкретність — посилання на статті, а не загальні слова]
2. [правильний стиль — офіційний/дружній/технічний]
3. [повнота — кроки, строки, умови]
4. [обробка edge cases — що якщо ситуація нестандартна]

### Що робить датасет ПОГАНИМ:
1. [відсутність конкретних норм]
2. [неправильна юрисдикція або застарілі дані]
3. [занадто загальний стиль]

### Мій ідеальний приклад виглядає так:
Input: "..."
Output: "..."

### Ціль: 2000 таких прикладів
```

---

### День 4–5: «Отруєний датасет» (16 годин)

**Мета:** Побачити як 5% поганих прикладів знищують якість всього датасету. Розвинути «нюх» на поганий приклад.

```python
# experiments/09_poisoned_dataset.py
"""
Візуалізувати ефект отруєних прикладів на якість.
Метод: LLM-judge оцінює «модель» натреновану на чистих vs забруднених даних.
"""
import json
import random
from openai import OpenAI
from pathlib import Path

client = OpenAI()

# ── Чистий датасет (30 хороших прикладів) ─────────────────────────────────
CLEAN_EXAMPLES = [
    {
        "q": "Роботодавець затримав зарплату на 15 днів. Що я маю право вимагати?",
        "a": """**Правова норма:** ст. 116, 117 КЗпП України; ст. 34 ЗУ "Про оплату праці"
**Суть:** При затримці більше 7 днів — нараховується компенсація.
**Ваші дії:**
1. Написати письмову претензію роботодавцю
2. Вимагати виплату зарплати + індексацію (ст. 34 ЗУ)
3. Якщо відмова — звернутись до Держпраці або суду
**Строки:** Позов до суду — в межах 3 місяців
**Рівень впевненості:** Впевнений ✅"""
    },
    # ... в реальному файлі тут 29 ще прикладів
]

# ── Отруєні приклади (галюцинації, неправильні поради) ────────────────────
POISON_EXAMPLES = [
    {
        "q": "Скільки днів відпустки?",
        "a": "Зазвичай 14 днів. Але деякі компанії дають більше якщо попросиш нічого."
        # ❌ Неправильно: закон каже 24 дні, без "якщо попросиш"
    },
    {
        "q": "Мене звільнили. Що робити?",
        "a": "Нічого не можна зробити після підписання наказу. Треба шукати нову роботу."
        # ❌ Неправильно: є право оскарження в суді протягом 30 днів!
    },
    {
        "q": "Роботодавець не платить зарплату 2 місяці.",
        "a": "Можна піти в поліцію і написати заяву про крадіжку."
        # ❌ Неправильно: це Держпраця або суд, а не поліція
    },
]

def simulate_model_with_dataset(examples: list, question: str) -> str:
    """Симулюємо fine-tune через few-shot з прикладами датасету."""
    msgs = [{"role": "system", "content": "Ти юридичний асистент по трудовому праву України."}]
    for ex in examples[:5]:  # Беремо перші 5 як few-shot
        msgs.append({"role": "user",      "content": ex["q"]})
        msgs.append({"role": "assistant", "content": ex["a"]})
    msgs.append({"role": "user", "content": question})
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.1
    )
    return resp.choices[0].message.content

async def judge_answer(question: str, answer: str) -> dict:
    """LLM-judge: оцінити якість відповіді."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """Оціни юридичну відповідь від 1 до 10.
Повертай JSON: {"score": N, "is_dangerous": bool, "reason": "..."}
is_dangerous=true якщо відповідь може нашкодити користувачу."""},
            {"role": "user", "content": f"Питання: {question}\nВідповідь: {answer}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

# Тестові питання
TEST_QS = [
    "Скільки днів щорічної відпустки за законом?",
    "Мене звільнили незаконно. Що робити?",
    "Затримали зарплату. Як отримати компенсацію?",
]

print("🧪 ТЕСТ: Чистий датасет vs Отруєний\n")

for q in TEST_QS:
    print(f"\n{'='*60}")
    print(f"ПИТАННЯ: {q}")
    
    # Модель 1: тільки чисті приклади
    clean_answer = simulate_model_with_dataset(CLEAN_EXAMPLES, q)
    
    # Модель 2: 95% чисті + 5% отруєних (реалістична пропорція)
    poisoned_set = CLEAN_EXAMPLES.copy()
    poisoned_set.extend(POISON_EXAMPLES)
    random.shuffle(poisoned_set)
    poisoned_answer = simulate_model_with_dataset(poisoned_set, q)
    
    print(f"\n✅ ЧИСТИЙ датасет:")
    print(f"   {clean_answer[:300]}...")
    
    print(f"\n☠️  ОТРУЄНИЙ датасет:")
    print(f"   {poisoned_answer[:300]}...")

print("\n\n📊 ВИСНОВОК:")
print("Навіть 5% поганих прикладів можуть:")
print("  - Змінити стиль відповідей")
print("  - Внести неправильні факти")
print("  - Зробити відповіді небезпечними для користувачів")
print("\nНаступний крок: навчитись ВИЯВЛЯТИ погані приклади автоматично")
```

---

## ТИЖДЕНЬ 2: ТЕХНІКИ ЗБОРУ ДАНИХ

### Техніка 1: «Python-стек для даних» (24 години)

**Проблема яку вирішує:** Збір 2000 якісних прикладів вручну займе 200+ годин. Треба автоматизація: паралельні API виклики, очищення, JSONL pipeline.

```python
# experiments/10_data_pipeline.py
"""
Production-ready pipeline збору даних:
- asyncio для паралельних API викликів (10x швидше)
- JSONL формат з метаданими
- Автоматичне збереження прогресу (resume після краша)
- Дедублікація на льоту
"""
import asyncio
import json
import os
import hashlib
import time
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI()

# ══════════════════════════════════════════════════════════════
# НАЛАШТУЙ ПІД СВОЮ НІШУ
NICHE = "трудове право України"
SEED_QUESTIONS = [
    "Затримка виплати заробітної плати",
    "Незаконне звільнення",
    "Права при скороченні штату",
    "Відпустки та їх оплата",
    "Декретна відпустка і захист вагітних",
    "Лікарняні і виплати по непрацездатності",
    "Випробувальний термін: права і обмеження",
    "Понаднормова робота і компенсація",
    "Переведення на іншу посаду без згоди",
    "Матеріальна відповідальність працівника",
]
# ══════════════════════════════════════════════════════════════

SEMAPHORE = asyncio.Semaphore(5)  # максимум 5 паралельних запитів
PROGRESS_FILE = Path("data/collection_progress.jsonl")
SEEN_HASHES: set[str] = set()

def text_hash(text: str) -> str:
    """Коротка сигнатура тексту для дедублікації."""
    return hashlib.md5(text.lower().strip()[:200].encode()).hexdigest()[:12]

def load_progress() -> int:
    """Завантажити вже зібрані приклади, повернути кількість."""
    if not PROGRESS_FILE.exists():
        return 0
    count = 0
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
                SEEN_HASHES.add(text_hash(ex["input"]))
                count += 1
            except:
                pass
    return count

async def generate_qa_batch(topic: str, batch_size: int = 5) -> list[dict]:
    """Генерувати batch питань і відповідей для теми."""
    async with SEMAPHORE:
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""Ти генератор навчальних даних для {NICHE}.
Генеруй різноманітні пари питання-відповідь.
Відповіді мають бути: конкретними, з посиланням на статті КЗпП, практичними.
Повертай JSON масив: [{{"input": "питання", "output": "відповідь"}}]"""},
                    {"role": "user", "content": f"Тема: {topic}\nЗгенеруй {batch_size} різних пар QA."}
                ],
                response_format={"type": "json_object"},
                temperature=0.8  # висока температура = більша різноманітність
            )
            data = json.loads(resp.choices[0].message.content)
            # API може повернути {"items": [...]} або {"data": [...]} або [...]
            if isinstance(data, list):
                return data
            for key in ["items", "data", "examples", "pairs"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return []
        except Exception as e:
            print(f"  ⚠️  Помилка для теми '{topic}': {e}")
            return []

async def collect_dataset(target_size: int = 2000):
    """Зібрати датасет заданого розміру."""
    Path("data").mkdir(exist_ok=True)
    
    already_collected = load_progress()
    print(f"📦 Вже зібрано: {already_collected} прикладів")
    
    if already_collected >= target_size:
        print(f"✅ Ціль {target_size} вже досягнута!")
        return
    
    needed = target_size - already_collected
    print(f"🎯 Потрібно ще: {needed} прикладів\n")
    
    # Рахуємо скільки батчів треба
    batches_per_topic = max(1, needed // (len(SEED_QUESTIONS) * 5))
    
    tasks = []
    for topic in SEED_QUESTIONS:
        for _ in range(batches_per_topic):
            tasks.append(generate_qa_batch(topic, batch_size=5))
    
    collected = already_collected
    duplicates = 0
    
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        async for batch in tqdm_asyncio.as_completed(
            tasks, desc="Збір даних", total=len(tasks)
        ):
            for item in batch:
                if not item.get("input") or not item.get("output"):
                    continue
                
                h = text_hash(item["input"])
                if h in SEEN_HASHES:
                    duplicates += 1
                    continue
                
                SEEN_HASHES.add(h)
                record = {
                    "input":  item["input"].strip(),
                    "output": item["output"].strip(),
                    "source": "gpt4o-mini-generated",
                    "topic":  "unknown",  # буде заповнено в post-processing
                    "ts":     int(time.time()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected += 1
                
                if collected >= target_size:
                    break
    
    print(f"\n✅ Зібрано: {collected} прикладів")
    print(f"🔁 Дублів відфільтровано: {duplicates}")
    print(f"💾 Збережено: {PROGRESS_FILE}")

if __name__ == "__main__":
    asyncio.run(collect_dataset(target_size=2000))
```

**Завдання після запуску:**
1. Переглянь 50 випадкових прикладів — які погані? Чому?
2. Додай перевірку: чи є в кожній відповіді посилання на закон?
3. Напиши фільтр що автоматично відкидає відповіді без `КЗпП` або `ЗУ`

---

### Техніка 2: «HuggingFace Datasets Ecosystem» (16 годин)

**Проблема яку вирішує:** Твій датасет не може бути тільки з GPT-4o — ризик distribution collapse. Треба різноманітні джерела.

```python
# experiments/11_huggingface_datasets.py
"""
Знайти, завантажити і злити публічні датасети.
Мета: додати до синтетичних даних реальні людські питання.
"""
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import json
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# Знайти датасети: https://huggingface.co/datasets
# Для своєї ніші шукай: "legal qa", "ukrainian", "law"
# Замни на реальні датасети що знайдеш
# ══════════════════════════════════════════════════════════════

def explore_dataset(dataset_name: str, subset: str = None, split: str = "train"):
    """Розвідка датасету: що в ньому є."""
    print(f"\n{'='*60}")
    print(f"📦 Датасет: {dataset_name}")
    
    try:
        ds = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
        print(f"   Розмір: {len(ds)} прикладів")
        print(f"   Колонки: {ds.column_names}")
        print(f"   Перший приклад:")
        first = ds[0]
        for k, v in first.items():
            val = str(v)[:100]
            print(f"     {k}: {val}")
        return ds
    except Exception as e:
        print(f"   ❌ Помилка: {e}")
        return None

def convert_to_qa_format(ds, input_col: str, output_col: str) -> list[dict]:
    """Конвертувати датасет в єдиний формат {input, output}."""
    result = []
    for item in ds:
        inp = str(item.get(input_col, "")).strip()
        out = str(item.get(output_col, "")).strip()
        if inp and out and len(inp) > 20 and len(out) > 50:
            result.append({
                "input":  inp,
                "output": out,
                "source": f"huggingface/{ds.info.dataset_name if hasattr(ds, 'info') else 'unknown'}",
            })
    return result

# Спроба завантажити публічні датасети
# ЗАМНИ НА ТІ ЩО ЗНАЙШОВ ДЛЯ СВОЄЇ НІШІ
CANDIDATE_DATASETS = [
    # ("legal-bert-base-uncased", None),   # English legal
    # ("joelniklaus/legalbench", "abercrombie"),  # Legal reasoning
    # Для перевірки — загальний QA датасет
    ("squad", None),  # для демонстрації формату
]

all_external = []
for name, subset in CANDIDATE_DATASETS:
    ds = explore_dataset(name, subset)
    if ds is None:
        continue
    
    # Потрібно адаптувати колонки під свій датасет
    # Squad має: id, title, context, question, answers
    # Твій датасет може мати інші колонки — визнач вручну
    print(f"   → Потрібна ручна адаптація колонок для цього датасету")

# ── Фільтрація і очищення ──────────────────────────────────────────────────
def clean_dataset(examples: list[dict]) -> list[dict]:
    """Відфільтрувати погані приклади."""
    clean = []
    for ex in examples:
        inp, out = ex.get("input", ""), ex.get("output", "")
        
        # Фільтри якості
        if len(inp) < 20:          continue  # Занадто короткий вхід
        if len(out) < 50:          continue  # Занадто коротка відповідь
        if len(out) > 3000:        continue  # Занадто довга відповідь
        if out.count("...") > 3:   continue  # Обрізаний текст
        if "Example:" in out[:50]: continue  # Метакоментар замість відповіді
        
        clean.append(ex)
    
    print(f"Відфільтровано: {len(examples)} → {len(clean)} ({len(examples)-len(clean)} видалено)")
    return clean

# ── Злиття джерел ──────────────────────────────────────────────────────────
def merge_sources(*datasets: list) -> list[dict]:
    """Злити декілька датасетів в один."""
    merged = []
    for ds in datasets:
        merged.extend(ds)
    print(f"\nЗлиття: {[len(d) for d in datasets]} → {len(merged)} разом")
    return merged

# ── Статистика датасету ─────────────────────────────────────────────────────
def dataset_stats(examples: list[dict]):
    """Вивести статистику датасету."""
    if not examples:
        print("Датасет порожній!")
        return
    
    input_lens  = [len(ex["input"])  for ex in examples]
    output_lens = [len(ex["output"]) for ex in examples]
    sources     = {}
    for ex in examples:
        src = ex.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\n📊 СТАТИСТИКА ДАТАСЕТУ")
    print(f"   Всього: {len(examples)} прикладів")
    print(f"   Input:  avg={sum(input_lens)//len(input_lens)} chars, "
          f"min={min(input_lens)}, max={max(input_lens)}")
    print(f"   Output: avg={sum(output_lens)//len(output_lens)} chars, "
          f"min={min(output_lens)}, max={max(output_lens)}")
    print(f"\n   Джерела:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"     {src}: {cnt} ({cnt/len(examples)*100:.1f}%)")

# Приклад використання
synthetic_data = []  # завантаж з data/collection_progress.jsonl
if Path("data/collection_progress.jsonl").exists():
    with open("data/collection_progress.jsonl", encoding="utf-8") as f:
        synthetic_data = [json.loads(l) for l in f if l.strip()]

clean_synthetic = clean_dataset(synthetic_data)
dataset_stats(clean_synthetic)
```

---

## ТИЖДЕНЬ 3: РОЗШИРЕННЯ І ЯКІСТЬ

### Техніка 3: «Self-Instruct Pipeline» (24 години)

**Проблема яку вирішує:** 2000 прикладів потрібні, але вручну написати складно. Self-instruct: 50 seed-прикладів → GPT-4o генерує 1000+ різноманітних варіацій.

**Принцип:** Якщо seed-приклади якісні → генеровані теж якісні. Garbage in = garbage out.

```python
# experiments/12_self_instruct.py
"""
Self-Instruct: розширення датасету через GPT-4o.

Алгоритм:
1. Маємо 50 seed прикладів (написані вручну — золотий стандарт)
2. Для кожного seed — генеруємо 10 варіацій
3. Перевіряємо diversity: відкидаємо дублі через embedding similarity
4. Результат: 50 → ~400 унікальних прикладів
"""
import asyncio
import json
import random
from pathlib import Path
from openai import AsyncOpenAI

client = AsyncOpenAI()
SEMAPHORE = asyncio.Semaphore(5)

# ── 50 seed прикладів (написати ВРУЧНУ — це твій золотий стандарт) ────────
# Для демонстрації — 5 прикладів. В реальному проекті — 50.
SEED_EXAMPLES = [
    {
        "input": "Мене скорочують. Яку компенсацію я маю отримати?",
        "output": """**Правова норма:** ст. 44 КЗпП України
**Вихідна допомога:** мінімум 1 середньомісячна зарплата
**Обов'язково також:**
1. Виплата за всі невикористані дні відпустки (ст. 83 КЗпП)
2. Повний розрахунок в день звільнення (ст. 116 КЗпП)
3. Попередження за 2 місяці — письмово (ст. 49-2 КЗпП)
**Якщо не виплатили:** Держпраця → суд → стягнення + штраф 3x
**Рівень впевненості:** Впевнений ✅"""
    },
    {
        "input": "Роботодавець змушує підписати нову угоду з нижчою зарплатою.",
        "output": """**Правова норма:** ст. 32 КЗпП України
**Суть:** Зміна умов праці — лише за письмовою згодою працівника
**Ваші права:**
1. Відмовитись підписувати без наслідків для роботи
2. Вимагати зберегти поточні умови
3. Якщо тиснуть — зафіксувати (листи, записи) і звернутись до Держпраці
**Ризик:** Роботодавець може запустити процедуру скорочення — але тоді зобов'язаний виплатити компенсацію
**Рівень впевненості:** Впевнений ✅"""
    },
    {
        "input": "Можна відмовитись від надурочної роботи?",
        "output": """**Правова норма:** ст. 62-65 КЗпП України
**Відповідь:** Так, в більшості випадків.
**Понаднормова робота дозволена лише:**
- За письмовою згодою працівника (загальний випадок)
- Без згоди: аварія, стихійне лихо, усунення загрози нещасного випадку
**Обмеження:** Не більше 4 год протягом 2 днів і 120 год на рік
**Оплата:** Мінімум 2x тариф за кожну понаднормову годину
**Рівень впевненості:** Впевнений ✅"""
    },
]

async def generate_variations(seed: dict, n: int = 8) -> list[dict]:
    """Згенерувати N варіацій seed-прикладу."""
    async with SEMAPHORE:
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Ти генератор навчальних даних для системи юридичних консультацій.
На основі прикладу генеруй різноманітні варіації:
- Різні формулювання того самого питання
- Пов'язані але відмінні ситуації  
- Різний контекст (різний стаж, різна зарплата тощо)
НЕ копіюй приклад дослівно. Кожна варіація — нова унікальна ситуація.
Повертай JSON: {"variations": [{"input": "...", "output": "..."}, ...]}"""},
                    {"role": "user", "content": f"""Приклад:
Input: {seed['input']}
Output: {seed['output']}

Згенеруй {n} різноманітних варіацій цього типу питання."""}
                ],
                response_format={"type": "json_object"},
                temperature=0.9
            )
            data = json.loads(resp.choices[0].message.content)
            variations = data.get("variations", [])
            # Додати метадані
            for v in variations:
                v["source"] = "self-instruct"
                v["seed_input"] = seed["input"][:50]
            return variations
        except Exception as e:
            print(f"  ⚠️  Помилка: {e}")
            return []

async def run_self_instruct(seeds: list[dict], variations_per_seed: int = 8):
    """Запустити повний self-instruct pipeline."""
    print(f"🌱 Seeds: {len(seeds)} прикладів")
    print(f"🎯 Ціль: ~{len(seeds) * variations_per_seed} варіацій\n")
    
    tasks = [generate_variations(seed, variations_per_seed) for seed in seeds]
    
    all_results = [seed.copy() for seed in seeds]  # включаємо self seeds
    
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        batch = await coro
        all_results.extend(batch)
        print(f"  [{i+1}/{len(seeds)}] +{len(batch)} варіацій → всього {len(all_results)}")
    
    # Зберегти
    Path("data").mkdir(exist_ok=True)
    out_path = Path("data/self_instruct_raw.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_results:
            if item.get("input") and item.get("output"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Збережено: {len(all_results)} прикладів → {out_path}")
    return all_results

if __name__ == "__main__":
    results = asyncio.run(run_self_instruct(SEED_EXAMPLES, variations_per_seed=8))
    
    # Перевірити diversity
    print("\n📊 Перші 5 варіацій:")
    for r in results[len(SEED_EXAMPLES):len(SEED_EXAMPLES)+5]:
        print(f"\n  Q: {r['input'][:80]}")
        print(f"  A: {r['output'][:150]}...")
```

---

### Техніка 4: «Semantic Deduplication» (16 годин)

**Проблема яку вирішує:** Self-instruct генерує дублі. "Скільки днів відпустки?" і "Який мінімум відпускних днів?" — це одне і те ж. Fine-tuning на дублях = переучування.

```python
# experiments/13_semantic_dedup.py
"""
Семантична дедублікація через embeddings.

Алгоритм:
1. Отримати embedding кожного input через OpenAI
2. Порахувати cosine similarity між усіма парами
3. Якщо similarity > 0.92 — залишити тільки один (кращий)
4. Виміряти: наскільки датасет "різноманітний" до і після
"""
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Отримати embeddings батчами."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small",  # дешевший варіант
            input=batch
        )
        batch_embeddings = [e.embedding for e in resp.data]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings, dtype=np.float32)

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Матриця cosine similarity. Нормалізуємо для швидкості."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)
    return normalized @ normalized.T

def greedy_dedup(
    examples: list[dict],
    embeddings: np.ndarray,
    threshold: float = 0.92
) -> tuple[list[dict], dict]:
    """
    Greedy deduplication:
    Проходимо по всіх прикладах.
    Якщо поточний приклад схожий (> threshold) на вже збережений — пропускаємо.
    """
    n = len(examples)
    sim_matrix = cosine_similarity_matrix(embeddings)
    
    kept_indices = []
    removed = 0
    
    for i in tqdm(range(n), desc="Dedup"):
        is_duplicate = False
        for j in kept_indices:
            if sim_matrix[i, j] > threshold:
                is_duplicate = True
                removed += 1
                break
        if not is_duplicate:
            kept_indices.append(i)
    
    kept = [examples[i] for i in kept_indices]
    
    stats = {
        "original": n,
        "kept": len(kept),
        "removed": removed,
        "removal_rate": f"{removed/n*100:.1f}%",
        "threshold": threshold,
    }
    
    return kept, stats

def measure_diversity(embeddings: np.ndarray) -> dict:
    """
    Виміряти різноманітність датасету:
    - Середня попарна відстань (вище = різноманітніший)
    - Мінімальна відстань (нижче = є пари-близнюки)
    """
    sim = cosine_similarity_matrix(embeddings)
    # Беремо лише upper triangle (без діагоналі)
    upper = sim[np.triu_indices(len(embeddings), k=1)]
    
    return {
        "mean_similarity":  float(np.mean(upper)),
        "max_similarity":   float(np.max(upper)),
        "min_similarity":   float(np.min(upper)),
        "near_dups_092":    int(np.sum(upper > 0.92)),
        "near_dups_085":    int(np.sum(upper > 0.85)),
    }

# ── Запуск ─────────────────────────────────────────────────────────────────
data_path = Path("data/self_instruct_raw.jsonl")
if not data_path.exists():
    print("⚠️  Спочатку запусти 12_self_instruct.py")
    exit()

examples = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
print(f"📦 Завантажено: {len(examples)} прикладів")

# Embeddings тільки для inputs (коротші → дешевші)
inputs = [ex["input"] for ex in examples]
print(f"\n1️⃣  Отримуємо embeddings...")
embeddings = get_embeddings_batch(inputs)

print(f"\n2️⃣  Різноманітність ДО дедуплікації:")
before_stats = measure_diversity(embeddings)
for k, v in before_stats.items():
    print(f"   {k}: {v}")

print(f"\n3️⃣  Запускаємо greedy dedup (threshold=0.92)...")
deduped, dedup_stats = greedy_dedup(examples, embeddings, threshold=0.92)
print(f"\n   Результат: {dedup_stats}")

# Embeddings для дедуплікованого датасету
deduped_inputs = [ex["input"] for ex in deduped]
deduped_embs = get_embeddings_batch(deduped_inputs)

print(f"\n4️⃣  Різноманітність ПІСЛЯ дедуплікації:")
after_stats = measure_diversity(deduped_embs)
for k, v in after_stats.items():
    print(f"   {k}: {v}")

# Зберегти дедуплікований датасет
out_path = Path("data/dataset_deduped.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for ex in deduped:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n✅ Збережено: {len(deduped)} прикладів → {out_path}")
print(f"\n📊 Якість датасету зросла:")
print(f"   Середня схожість: {before_stats['mean_similarity']:.3f} → {after_stats['mean_similarity']:.3f}")
print(f"   Майже-дублі (>0.92): {before_stats['near_dups_092']} → {after_stats['near_dups_092']}")
```

---

## ТИЖДЕНЬ 4: ЯКІСТЬ І РОУТЕР

### Техніка 5: «LLM-Judge Калібрація» (16 годин)

**Проблема яку вирішує:** LLM-judge може системно помилятись — надто м'яко оцінювати, ігнорувати певні категорії помилок, мати positivity bias. Треба перевірити correlation з людиною.

```python
# experiments/14_judge_calibration.py
"""
Перевірити наскільки LLM-judge узгоджується з людиною.
Метрика: Cohen's kappa (>0.6 = добра узгодженість)

Процес:
1. Беремо 50 прикладів з датасету
2. Людина (ти) оцінює кожну відповідь: 1-5
3. LLM-judge оцінює: 1-5
4. Рахуємо Cohen's kappa
5. Знаходимо патерни де judge помиляється
"""
import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from collections import Counter

client = AsyncOpenAI()

JUDGE_PROMPT = """Ти оцінювач якості юридичних консультацій по трудовому праву України.

Оціни відповідь за шкалою 1-5:
1 = Небезпечно неправильно або марно
2 = Суттєві помилки або прогалини
3 = Частково правильно, є недоліки
4 = Добра відповідь, невеликі недоліки
5 = Відмінна відповідь, точна і практична

Критерії оцінювання:
- Посилання на конкретні статті законів (КЗпП, ЗУ)
- Практичність рекомендацій
- Відсутність галюцинацій
- Правильна юрисдикція (Україна, 2024-2025)

Відповідай JSON: {"score": N, "reason": "...", "issues": ["..."]}"""

async def judge_example(q: str, a: str) -> dict:
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",   "content": f"Питання: {q}\nВідповідь: {a}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(resp.choices[0].message.content)

def cohens_kappa(human_scores: list[int], llm_scores: list[int]) -> float:
    """Розрахунок Cohen's kappa."""
    n = len(human_scores)
    assert n == len(llm_scores)
    
    # Observed agreement
    p_o = sum(1 for h, l in zip(human_scores, llm_scores) if h == l) / n
    
    # Expected agreement
    human_dist  = Counter(human_scores)
    llm_dist    = Counter(llm_scores)
    categories  = set(human_scores) | set(llm_scores)
    
    p_e = sum(
        (human_dist.get(c, 0) / n) * (llm_dist.get(c, 0) / n)
        for c in categories
    )
    
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)

async def run_calibration():
    # Завантажити 50 прикладів
    examples = []
    data_path = Path("data/dataset_deduped.jsonl")
    if data_path.exists():
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50: break
                examples.append(json.loads(line))
    
    if not examples:
        print("⚠️  Спочатку запусти 13_semantic_dedup.py")
        return
    
    print(f"📋 Оцінюємо {len(examples)} прикладів через LLM-judge...")
    
    # LLM оцінки (паралельно)
    tasks = [judge_example(ex["input"], ex["output"]) for ex in examples]
    llm_judgments = await asyncio.gather(*tasks)
    llm_scores = [j.get("score", 3) for j in llm_judgments]
    
    # Зберегти для ручного огляду
    calibration_data = []
    for ex, judg in zip(examples, llm_judgments):
        calibration_data.append({
            "input":       ex["input"],
            "output":      ex["output"][:300],
            "llm_score":   judg.get("score"),
            "llm_reason":  judg.get("reason", ""),
            "llm_issues":  judg.get("issues", []),
            "human_score": None,  # ← ЗАПОВНИ ВРУЧНУ
        })
    
    out_path = Path("data/calibration_for_human.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in calibration_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Збережено: {out_path}")
    print(f"\n📝 ЗАВДАННЯ (2-3 години):")
    print("Відкрий файл і для кожного прикладу заповни human_score (1-5)")
    print("Потім запусти: python 14_judge_calibration.py --analyze")
    
    # Аналіз після ручного заповнення
    print(f"\nРозподіл LLM scores:")
    score_counts = Counter(llm_scores)
    for score in sorted(score_counts):
        bar = "█" * score_counts[score]
        print(f"  {score}/5: {bar} ({score_counts[score]})")
    
    print(f"\nСередній LLM score: {sum(llm_scores)/len(llm_scores):.2f}")
    print("\nОзнаки positivity bias якщо середній > 4.0 — judge занадто м'який!")

if __name__ == "__main__":
    asyncio.run(run_calibration())
```

---

### Техніка 6: «Production Cost-Quality Router» (24 години)

> **Це повна версія Cost Router з Білого поясу.** Там був навчальний прототип. Тут — production-ready система з точним підрахунком вартості, статистикою routing, і eval для перевірки що routing не шкодить якості.

```python
# my_specialist/router.py
"""
Production Cost-Quality Router v2.0

Покращення від White Belt preview:
- tiktoken для точного підрахунку токенів
- Routing history і статистика
- A/B eval: compare routed vs all-GPT-4o
- Configurable threshold per task type
"""
import json
import time
import tiktoken
from pathlib import Path
from dataclasses import dataclass, field, asdict
from openai import OpenAI
import ollama

client = OpenAI()

# ── Токен-точна вартість ($/1M tokens, травень 2026) ────────────────────────
PRICING = {
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    "local":       {"input": 0.00,  "output": 0.00},
}

# ── Tiktoken для точного підрахунку ─────────────────────────────────────────
_encoders: dict = {}

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Точний підрахунок токенів через tiktoken."""
    enc_model = "gpt-4o" if "gpt-4o" in model else "gpt-3.5-turbo"
    if enc_model not in _encoders:
        _encoders[enc_model] = tiktoken.encoding_for_model(enc_model)
    return len(_encoders[enc_model].encode(text))

def calc_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Точна вартість виклику."""
    pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

@dataclass
class RouterStats:
    """Статистика routing для моніторингу."""
    total_requests: int = 0
    model_counts: dict = field(default_factory=lambda: {"local": 0, "gpt-4o-mini": 0, "gpt-4o": 0})
    total_cost_usd: float = 0.0
    hypothetical_cost_usd: float = 0.0  # якби завжди GPT-4o
    savings_usd: float = 0.0
    
    def update(self, model_used: str, cost: float, full_cost: float):
        self.total_requests += 1
        self.model_counts[model_used] = self.model_counts.get(model_used, 0) + 1
        self.total_cost_usd += cost
        self.hypothetical_cost_usd += full_cost
        self.savings_usd = self.hypothetical_cost_usd - self.total_cost_usd
    
    def report(self) -> str:
        if not self.total_requests:
            return "Немає даних"
        dist = {k: f"{v/self.total_requests*100:.0f}%" for k, v in self.model_counts.items()}
        return (
            f"Запитів: {self.total_requests} | "
            f"Routing: {dist} | "
            f"Витрати: ${self.total_cost_usd:.4f} | "
            f"Економія vs all-GPT4o: ${self.savings_usd:.4f} "
            f"({self.savings_usd/max(self.hypothetical_cost_usd,0.0001)*100:.0f}%)"
        )

_stats = RouterStats()

def classify_complexity(question: str, system_prompt: str = "") -> str:
    """
    Класифікація складності — надійна версія:
    - 1.5B модель (не 0.5B)
    - Системний промпт англійською (краща надійність)
    - startswith а не 'in'
    """
    try:
        resp = ollama.chat(
            model='qwen2.5:1.5b',
            messages=[
                {'role': 'system', 'content': (
                    "Classify the complexity of this user request. "
                    "Reply with ONLY one word: simple, medium, or complex.\n"
                    "simple = single fact, yes/no, direct lookup\n"
                    "medium = 2-3 aspects, some reasoning needed\n"
                    "complex = calculations, multiple scenarios, legal strategy"
                )},
                {'role': 'user', 'content': question}
            ]
        )
        word = resp['message']['content'].strip().lower().split()[0] \
               if resp['message']['content'].strip() else 'medium'
        if word.startswith('simple'):  return 'simple'
        if word.startswith('complex'): return 'complex'
        return 'medium'
    except Exception:
        return 'medium'  # безпечний дефолт при збої класифікатора

def routed_call(
    question: str,
    system_prompt: str,
    few_shot_msgs: list[dict] | None = None
) -> dict:
    """
    Зробити API виклик через оптимальну модель.
    
    Returns:
        dict з полями: answer, model, complexity, cost_usd, tokens
    """
    complexity  = classify_complexity(question, system_prompt)
    messages    = [{"role": "system", "content": system_prompt}]
    if few_shot_msgs:
        messages.extend(few_shot_msgs)
    messages.append({"role": "user", "content": question})
    
    # Підраховуємо input tokens для cost calculation
    input_text   = system_prompt + question + "".join(
        m.get("content","") for m in (few_shot_msgs or [])
    )
    input_tokens = count_tokens(input_text)
    
    # ── Вибір моделі ────────────────────────────────────────────────────────
    if complexity == 'simple':
        # Локальна модель — безкоштовно
        try:
            resp = ollama.chat(
                model='qwen2.5:7b',
                messages=[{'role': m.get('role','user'), 'content': m.get('content','')}
                          for m in messages]
            )
            answer       = resp['message']['content']
            output_tokens= count_tokens(answer)
            model_used   = "local"
            cost         = 0.0
        except Exception as e:
            print(f"  ⚠️  Ollama failed, fallback to gpt-4o-mini: {e}")
            complexity = 'medium'  # fallthrough
    
    if complexity == 'medium':
        resp         = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.1
        )
        answer       = resp.choices[0].message.content
        output_tokens= resp.usage.completion_tokens
        input_tokens = resp.usage.prompt_tokens
        model_used   = "gpt-4o-mini"
        cost         = calc_cost(input_tokens, output_tokens, "gpt-4o-mini")
    
    elif complexity == 'complex':
        resp         = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.1
        )
        answer       = resp.choices[0].message.content
        output_tokens= resp.usage.completion_tokens
        input_tokens = resp.usage.prompt_tokens
        model_used   = "gpt-4o"
        cost         = calc_cost(input_tokens, output_tokens, "gpt-4o")
    
    # Hypothetical cost якби завжди GPT-4o
    hypothetical = calc_cost(input_tokens, count_tokens(answer), "gpt-4o")
    _stats.update(model_used, cost, hypothetical)
    
    return {
        "answer":     answer,
        "model":      model_used,
        "complexity": complexity,
        "cost_usd":   cost,
        "tokens":     {"input": input_tokens, "output": output_tokens},
    }

def router_report() -> str:
    return _stats.report()

# ── Тест ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SYSTEM = "Ти юридичний асистент по трудовому праву України. Відповідай структуровано з посиланням на закон."
    
    test_cases = [
        ("Скільки днів відпустки?",                            "simple"),
        ("Мене звільнили під час лікарняного. Що робити?",     "medium"),
        ("Розрахуй компенсацію: стаж 7 р., зарплата 45000 грн, 22 дні невикористаної відпустки, скорочення", "complex"),
    ]
    
    print("🚦 PRODUCTION ROUTER TEST\n")
    for question, expected in test_cases:
        result = routed_call(question, SYSTEM)
        icon = "✅" if result["complexity"] == expected else "⚠️"
        print(f"{icon} [{result['complexity']:7s}] → {result['model']:12s} | "
              f"${result['cost_usd']:.5f} | {question[:55]}...")
    
    print(f"\n📊 {router_report()}")
```

---

## ТИЖДЕНЬ 5: ЗБІРКА ДАТАСЕТУ

### Фінальна збірка і форматування (32 години)

```python
# data_pipeline/build_final_dataset.py
"""
Зібрати фінальний датасет з усіх джерел.
Відформатувати під SFT (Supervised Fine-Tuning) chat template.
Розбити на train/val.
"""
import json
import random
from pathlib import Path
from collections import Counter

def load_all_sources() -> list[dict]:
    """Завантажити з усіх джерел."""
    sources = {
        "data/dataset_deduped.jsonl":     "self_instruct+hf",
        "data/collection_progress.jsonl": "gpt4o_generated",
        # "data/manual_examples.jsonl":   "manual",  # якщо є
    }
    
    all_examples = []
    for path, source_tag in sources.items():
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  Не знайдено: {path}")
            continue
        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                ex["source"] = ex.get("source", source_tag)
                all_examples.append(ex)
                count += 1
        print(f"  ✅ {path}: {count} прикладів")
    
    return all_examples

def format_for_sft(examples: list[dict], system_prompt: str) -> list[dict]:
    """
    Відформатувати під стандартний chat template для SFT.
    Формат: {"messages": [{"role": ..., "content": ...}, ...]}
    Сумісний з: Axolotl, Unsloth, LLaMA-Factory
    """
    formatted = []
    for ex in examples:
        formatted.append({
            "messages": [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]},
            ],
            "_meta": {
                "source": ex.get("source", "unknown"),
                "input_len":  len(ex["input"]),
                "output_len": len(ex["output"]),
            }
        })
    return formatted

def quality_filter(examples: list[dict]) -> list[dict]:
    """Фінальний фільтр якості перед збереженням."""
    clean = []
    issues = Counter()
    
    for ex in examples:
        msgs     = ex.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"),    "")
        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"),"")
        
        if len(user_msg) < 15:
            issues["too_short_input"] += 1; continue
        if len(asst_msg) < 60:
            issues["too_short_output"] += 1; continue
        if asst_msg.count("...") > 4:
            issues["truncated"] += 1; continue
        if "I cannot" in asst_msg[:50] or "As an AI" in asst_msg[:50]:
            issues["ai_refusal"] += 1; continue
        
        clean.append(ex)
    
    print(f"  Відфільтровано: {len(examples)} → {len(clean)}")
    if issues:
        print(f"  Причини видалення: {dict(issues)}")
    return clean

def train_val_split(
    examples: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list, list]:
    """Розбити на train/val."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]

# ── Запуск ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ти юридичний асистент що спеціалізується виключно на трудовому праві України.
Маєш глибокі знання КЗпП, ЗУ "Про оплату праці", ЗУ "Про відпустки".

ФОРМАТ: Правова норма → Пояснення → Ваші дії → Рівень впевненості
ЗАБОРОНЕНО: питання поза трудовим правом, вигадані статті, відповіді без норм."""

print("📦 ЗБІРКА ФІНАЛЬНОГО ДАТАСЕТУ\n")
print("Завантаження джерел:")
all_raw = load_all_sources()
print(f"\nВсього: {len(all_raw)} прикладів")

print("\nФорматування під SFT chat template...")
formatted = format_for_sft(all_raw, SYSTEM_PROMPT)

print("\nФінальний фільтр якості...")
filtered = quality_filter(formatted)

print(f"\nРозбивка train/val (90/10)...")
train, val = train_val_split(filtered, val_ratio=0.1)

# Збереження
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

for split_name, split_data in [("train_sft", train), ("val_sft", val)]:
    out_path = out_dir / f"{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in split_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  💾 {out_path}: {len(split_data)} прикладів")

print(f"\n✅ ГОТОВО!")
print(f"   Train: {len(train)} прикладів")
print(f"   Val:   {len(val)} прикладів")
print(f"   Разом: {len(filtered)} прикладів")

if len(filtered) >= 2000:
    print(f"\n🎯 Ціль Жовтого поясу досягнута: {len(filtered)} >= 2000 прикладів!")
else:
    remaining = 2000 - len(filtered)
    print(f"\n⚠️  Потрібно ще: {remaining} прикладів")
    print("   Запусти ще раз: 12_self_instruct.py або 10_data_pipeline.py")
```

---

## ТИЖДЕНЬ 6: АТЕСТАЦІЯ

### Eval датасету (16 годин)

```python
# evaluation/eval_dataset_quality.py
"""
Атестація Жовтого поясу:
1. LLM-judge оцінює 10% вибірки датасету
2. Перевірка semantic diversity
3. Документація джерел
4. Фінальний звіт
"""
import json
import asyncio
import random
from pathlib import Path
from openai import AsyncOpenAI
from collections import Counter

client = AsyncOpenAI()
SEMAPHORE = asyncio.Semaphore(5)

JUDGE_PROMPT = """Оціни якість прикладу навчального датасету по трудовому праву України.

Критерії:
- Юридична точність (правильні статті, актуальне законодавство)
- Практична корисність (конкретні кроки, а не загальні слова)
- Безпека (не надає небезпечних рекомендацій)
- Форматування (структуровано, читається)

Шкала: 1-10
Повертай JSON: {"score": N, "pass": bool, "issues": ["..."]}
pass=true якщо score >= 7"""

async def judge_sample(example: dict) -> dict:
    msgs = example.get("messages", [])
    q = next((m["content"] for m in msgs if m["role"] == "user"), "")
    a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    
    async with SEMAPHORE:
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user",   "content": f"Input: {q}\n\nOutput: {a}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
    result = json.loads(resp.choices[0].message.content)
    result["input_preview"]  = q[:80]
    result["output_preview"] = a[:120]
    return result

async def run_attestation():
    train_path = Path("data/train_sft.jsonl")
    if not train_path.exists():
        print("⚠️  Спочатку запусти build_final_dataset.py")
        return
    
    all_examples = [json.loads(l) for l in open(train_path, encoding="utf-8") if l.strip()]
    total = len(all_examples)
    
    # Оцінюємо 10% вибірки
    sample_size = max(50, int(total * 0.1))
    sample = random.sample(all_examples, min(sample_size, total))
    
    print(f"📊 АТЕСТАЦІЯ ЖОВТОГО ПОЯСУ")
    print(f"   Датасет: {total} прикладів")
    print(f"   Вибірка: {len(sample)} прикладів (10%)")
    print(f"\n   Запускаємо LLM-judge оцінювання...\n")
    
    tasks    = [judge_sample(ex) for ex in sample]
    results  = await asyncio.gather(*tasks)
    
    scores   = [r.get("score", 0) for r in results]
    passed   = [r for r in results if r.get("pass", False)]
    failed   = [r for r in results if not r.get("pass", False)]
    avg      = sum(scores) / len(scores)
    pass_rate = len(passed) / len(results) * 100
    
    all_issues = []
    for r in results:
        all_issues.extend(r.get("issues", []))
    top_issues = Counter(all_issues).most_common(5)
    
    print(f"{'='*60}")
    print(f"РЕЗУЛЬТАТИ АТЕСТАЦІЇ ЖОВТОГО ПОЯСУ")
    print(f"{'='*60}")
    print(f"📦 Датасет: {total} прикладів")
    print(f"📊 Середній LLM-judge score: {avg:.2f}/10")
    print(f"✅ Pass rate (>= 7): {pass_rate:.1f}% ({len(passed)}/{len(results)})")
    
    if top_issues:
        print(f"\n⚠️  Найчастіші проблеми:")
        for issue, cnt in top_issues:
            print(f"   {cnt}x {issue}")
    
    print(f"\n{'='*60}")
    if avg >= 8.0 and total >= 2000:
        print(f"✅ ЖОВТИЙ ПОЯС ОТРИМАНО!")
        print(f"   Score: {avg:.2f}/10 >= 8.0 ✅")
        print(f"   Розмір: {total} >= 2000 ✅")
        print(f"\n   Переходиш до Помаранчевого поясу — Fine-Tuning!")
    else:
        print(f"❌ Потрібно доопрацювання:")
        if avg < 8.0:
            print(f"   Score: {avg:.2f} < 8.0 — покращ якість прикладів")
        if total < 2000:
            print(f"   Розмір: {total} < 2000 — зберіть більше даних")
        
        if failed:
            print(f"\nПриклади що провалились:")
            for r in failed[:3]:
                print(f"  Score {r['score']}: {r['input_preview']}...")
                print(f"  Проблеми: {r.get('issues', [])}")
    
    # Зберегти звіт
    report = {
        "total_examples":  total,
        "sample_size":     len(sample),
        "avg_score":       avg,
        "pass_rate":       pass_rate,
        "passed":          len(passed),
        "failed":          len(failed),
        "top_issues":      [{"issue": i, "count": c} for i, c in top_issues],
        "attestation_pass": avg >= 8.0 and total >= 2000,
    }
    with open("evaluation/yellow_belt_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Звіт: evaluation/yellow_belt_report.json")

asyncio.run(run_attestation())
```

---

## СТРУКТУРА ПРОЕКТУ ЖОВТОГО ПОЯСУ

```
my_llm_project/
│
├── data/
│   ├── collection_progress.jsonl    ← автозбір через asyncio
│   ├── self_instruct_raw.jsonl      ← self-instruct варіації
│   ├── dataset_deduped.jsonl        ← після семантичного dedup
│   ├── train_sft.jsonl              ← фінальний train (2000+)
│   ├── val_sft.jsonl                ← val (10%)
│   ├── dataset_battle_results.json  ← порівняння датасетів (Тиждень 1)
│   └── dataset_comparison_report.md ← твій аналіз (заповниш вручну)
│
├── experiments/
│   ├── 08_dataset_battle.py
│   ├── 09_poisoned_dataset.py
│   ├── 10_data_pipeline.py
│   ├── 11_huggingface_datasets.py
│   ├── 12_self_instruct.py
│   ├── 13_semantic_dedup.py
│   └── 14_judge_calibration.py
│
├── my_specialist/
│   ├── specialist.py                ← з Білого поясу (не чіпаємо)
│   └── router.py                    ← Production router v2.0 (новий)
│
├── data_pipeline/
│   └── build_final_dataset.py       ← фінальна збірка
│
└── evaluation/
    ├── eval_gold.jsonl              ← 50 прикладів для атестації системи
    ├── calibration_for_human.jsonl  ← для ручної оцінки judge калібрації
    └── yellow_belt_report.json      ← фінальний звіт атестації
```

---

## КРИТЕРІЇ АТЕСТАЦІЇ ЖОВТОГО ПОЯСУ

### Обов'язкові

| # | Критерій | Як перевірити |
|---|---|---|
| 1 | `data/train_sft.jsonl` містить >= 2000 прикладів | `wc -l data/train_sft.jsonl` |
| 2 | LLM-judge score >= 8.0/10 на 10% вибірці | `python evaluation/eval_dataset_quality.py` |
| 3 | Семантична дедублікація пройдена (cosine < 0.92) | Лог з 13_semantic_dedup.py |
| 4 | Кожен приклад має поле `source` | Перевірити 10 рандомних записів |
| 5 | `dataset_comparison_report.md` заповнений | Наявність аналізу вручну |
| 6 | `router.py` запущений і показує економію > 40% | `python my_specialist/router.py` |

### Бажані

| # | Критерій | Ціль |
|---|---|---|
| 7 | >= 3 різних джерела даних | Diversity |
| 8 | Cohen's kappa judge vs human > 0.6 | Calibrated judge |
| 9 | Mean similarity після dedup < 0.65 | Dataset diversity |
| 10 | Pass rate (score >= 7) > 90% | High quality floor |

---

## ТЕОРІЯ (читати ПІСЛЯ виконання вправ)

### Чому якість даних важливіша за архітектуру

> *«Neural Scaling Laws» (Kaplan et al., 2020): при рівних обчисленнях — більший і чистіший датасет дає більше ніж більша модель.*

**Після Тижня 1 (Dataset Battle) стане зрозуміло:**
- Модель відтворює патерни з даних, а не «розуміє»
- 5% поганих прикладів = 5% шансів на небезпечну відповідь
- Стиль і формат передаються через дані, а не через промпти

### Tokenization і Chat Template (читати перед Тижнем 5)

Критично важливо для fine-tuning:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Chat template = як модель "бачить" твої дані
messages = [
    {"role": "system",    "content": "Ти юрист..."},
    {"role": "user",      "content": "Скільки відпустки?"},
    {"role": "assistant", "content": "24 дні..."},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
# <|im_start|>system\nТи юрист...<|im_end|>\n<|im_start|>user\n...
```

Якщо формат датасету не збігається з chat template моделі → fine-tune не навчить нічого корисного.

### Embedding Space і Semantic Similarity

Embeddings — вектори в N-вимірному просторі де схожі за змістом тексти близькі. `text-embedding-3-small` = 1536 вимірів. Cosine similarity = кут між векторами (0 = ортогональні, 1 = ідентичні).

Для датасету треба висока **diversity** — рівномірне покриття простору запитів, а не кластери навколо кількох тем.

### Ресурси (після 3 тижнів практики)

- **HuggingFace Course** (huggingface.co/course) — глави 3, 5: datasets і fine-tuning
- **"Scaling Data-Constrained Language Models"** (Muennighoff et al.) — наукова основа data quality
- **Karpathy "Let's build GPT"** (YouTube) — базовий PyTorch, токени, attention

---

## ТИПОВІ ПОМИЛКИ ЖОВТОГО ПОЯСУ

| Помилка | Симптом | Виправлення |
|---|---|---|
| Тільки GPT-4o як джерело | Distribution collapse, модель "звучить як ChatGPT" | Додати людські питання з форумів, Quora |
| Немає фільтрації | 20%+ прикладів з галюцинаціями або помилками | LLM-judge автофільтр + ручна перевірка 10% |
| Не перевіряти chat template | Fine-tune не навчає нічому | Перевірити формат до початку тренування |
| Self-instruct з малою температурою | Низька різноманітність, багато дублів | temperature=0.9+ для генерації |
| Ігнорувати довжину відповідей | Мікс дуже коротких і довгих → нестабільне навчання | Фільтр: 60-1500 chars output |
| Semantic dedup без перевірки | Видалено важливі теми | Перевірити coverage після dedup |
| Positivity bias судді | Всі приклади отримують 8-9 | Перевірити Cohen's kappa, додати негативні приклади |

---

*Жовтий пояс — це розуміння що **дані — це код**. Кожен поганий приклад — це баг у твоєму тренувальному наборі. Debugging датасету важливіший за debugging моделі.*
