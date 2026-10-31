"""Измеряет, что ломает каждый подсаженный баг по отдельности.

Скрипт берёт исправленный dz1-bughunt.ipynb, возвращает в код один баг или все три,
прогоняет проверки и обучающую ячейку ноутбука, затем считает метрики в режиме генерации:
батч из одного примера, на входе только прошлое.

Запуск из корня репозитория:
    python scripts/bug_effects.py
Результат: таблица в stdout и artifacts/bug_effects.json.
"""
import contextlib
import io
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / 'dz1-bughunt.ipynb'
OUTPUT = ROOT / 'artifacts' / 'bug_effects.json'

# Баг: (строки исправленного кода, строки кода с багом).
BUGS = {
    'pe': [('position = torch.arange(length, dtype=torch.float32).unsqueeze(1)',
            'position = torch.arange(1, length + 1, dtype=torch.float32).unsqueeze(1)')],
    'layernorm': [('mean = x.mean(dim=-1, keepdim=True)',
                   'mean = x.mean(dim=0, keepdim=True)'),
                  ('var = x.var(dim=-1, keepdim=True, unbiased=False)',
                   'var = x.var(dim=0, keepdim=True, unbiased=False)')],
    'softmax': [('weights = torch.softmax(scores, dim=-1)',
                 'weights = torch.softmax(scores, dim=-2)')],
}

VARIANTS = {
    'все баги исправлены': (),
    'только баг PE': ('pe',),
    'только баг LayerNorm': ('layernorm',),
    'только баг softmax': ('softmax',),
    'все три бага (исходный код)': ('pe', 'layernorm', 'softmax'),
}


def code_cell(needle: str) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    hits = [s for s in sources if needle in s]
    assert len(hits) == 1, f'{needle!r}: найдено {len(hits)} ячеек'
    return hits[0]


def load_module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, name, 'exec'), module.__dict__)
    return module


def load_variant(bugs: tuple[str, ...]) -> types.ModuleType:
    source = code_cell('class MiniGPT(nn.Module)')
    for bug in bugs:
        for fixed, broken in BUGS[bug]:
            assert source.count(fixed) == 1, f'строка не найдена: {fixed}'
            source = source.replace(fixed, broken)
    return load_module('variant_' + '_'.join(bugs or ('fixed',)), source)


def failed_checks(variant: types.ModuleType) -> list[str]:
    checks = load_module('checks', code_cell('CHECKS = []'))
    return [c['name'] for c, _ in checks.run(variant, verbose=False)]


@torch.no_grad()
def prefix_metrics(model, data, context_size: int, samples: int = 400) -> tuple[float, float]:
    """Потери и точность следующего символа так, как работает генерация."""
    model.eval()
    rng = torch.Generator().manual_seed(123)
    total, hits = 0.0, 0
    for _ in range(samples):
        length = int(torch.randint(1, context_size + 1, (1,), generator=rng))
        start = int(torch.randint(0, len(data) - length - 1, (1,), generator=rng))
        idx = data[start:start + length].unsqueeze(0)
        target = data[start + length]
        logits = model(idx)[0][0, -1]
        total += F.cross_entropy(logits, target).item()
        hits += int(logits.argmax() == target)
    return total / samples, hits / samples


@torch.no_grad()
def probes(model, data, vocab_size: int, context_size: int) -> dict:
    """Прямые замеры нарушенных свойств на обученной модели."""
    model.eval()
    window = data[:context_size].unsqueeze(0)

    # Сумма весов внимания по ключам: у распределения она равна единице.
    row_sums = model(window)[2][0].sum(dim=-1)

    # Меняем только последний токен: логиты на прошлых позициях обязаны остаться прежними.
    changed = window.clone()
    changed[0, -1] = (changed[0, -1] + 1) % vocab_size
    leak = (model(window)[0] - model(changed)[0])[:, :-1].abs().max().item()

    # Меняем весь вход: при батче 1 логиты обязаны реагировать на контекст.
    other = (window + 1) % vocab_size
    sensitivity = (model(window)[0] - model(other)[0]).abs().max().item()

    return dict(row_sum_min=row_sums.min().item(), row_sum_max=row_sums.max().item(),
                future_leak=leak, input_sensitivity=sensitivity)


def valid_bigrams(sample: str, text: str) -> float:
    """Доля соседних пар символов образца, которые встречаются в корпусе."""
    pairs = [sample[i:i + 2] for i in range(len(sample) - 1)]
    return sum(pair in text for pair in pairs) / len(pairs)


def marginal_entropy(data) -> float:
    """Энтропия частот символов: потери модели, которая не использует контекст."""
    counts = torch.bincount(data).float()
    probs = counts[counts > 0] / counts.sum()
    return -(probs * probs.log()).sum().item()


def swapped_pe_loss(model, data, config) -> float:
    """Потери, если веса обучены на верной таблице, а применяются с таблицей с багом."""
    broken = load_variant(('pe',)).sinusoidal_positional_encoding(config.context_size,
                                                                  config.d_model)
    model.pos_emb.copy_(broken)
    return prefix_metrics(model, data, config.context_size)[0]


def run_variant(bugs: tuple[str, ...]) -> dict:
    variant = load_variant(bugs)
    failed = failed_checks(variant)

    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code_cell('history = []'), 'training', 'exec'), variant.__dict__)
    model, data, text, history = (variant.__dict__[k] for k in ('model', 'data', 'text', 'history'))
    stoi, itos = variant.__dict__['stoi'], variant.__dict__['itos']
    config = variant.__dict__['cfg']

    loss, accuracy = prefix_metrics(model, data, config.context_size)
    torch.manual_seed(0)
    start = torch.tensor([[stoi['а']]])
    out = variant.generate(model, start, max_new_tokens=200, temperature=0.8, top_k=10)
    sample = ''.join(itos[int(i)] for i in out[0])

    result = dict(failed_checks=failed,
                  train_loss_first=history[0],
                  train_loss_last50=sum(history[-50:]) / 50,
                  generation_loss=loss,
                  generation_accuracy=accuracy,
                  valid_bigrams=valid_bigrams(sample, text),
                  vocab_size=config.vocab_size,
                  marginal_entropy=marginal_entropy(data),
                  sample=sample,
                  **probes(model, data, config.vocab_size, config.context_size))
    if not bugs:
        result['swapped_pe_generation_loss'] = swapped_pe_loss(model, data, config)
    return result


def positional_shift() -> dict:
    """Насколько таблица с багом совпадает с верной таблицей, сдвинутой на одну позицию."""
    fixed = load_variant(()).sinusoidal_positional_encoding(65, 64)
    broken = load_variant(('pe',)).sinusoidal_positional_encoding(64, 64)
    return dict(diff_to_shifted=(broken - fixed[1:]).abs().max().item(),
                diff_to_formula=(broken - fixed[:-1]).abs().max().item(),
                first_row_broken=broken[0, :4].tolist(),
                first_row_fixed=fixed[0, :4].tolist())


def main() -> None:
    results = {name: run_variant(bugs) for name, bugs in VARIANTS.items()}
    report = dict(variants=results, positional_shift=positional_shift())
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    header = ('вариант', 'провал', 'обуч.', 'генер.', 'точн.', 'биграммы',
              'строки min', 'строки max', 'утечка', 'реакция')
    print(' | '.join(header))
    for name, r in results.items():
        print(f"{name} | {len(r['failed_checks'])} | {r['train_loss_last50']:.3f} | "
              f"{r['generation_loss']:.3f} | {r['generation_accuracy']:.2f} | "
              f"{r['valid_bigrams']:.2f} | {r['row_sum_min']:.3f} | {r['row_sum_max']:.3f} | "
              f"{r['future_leak']:.2e} | {r['input_sensitivity']:.2e}")
    for name, r in results.items():
        print(f"\n{name}\n  проваленные проверки: {', '.join(r['failed_checks']) or 'нет'}"
              f"\n  образец: {r['sample']!r}")
    print('\nсдвиг PE:', report['positional_shift'])
    fixed = results['все баги исправлены']
    print('верные веса с таблицей с багом, потери в режиме генерации:',
          fixed['swapped_pe_generation_loss'])
    print('размер словаря:', fixed['vocab_size'],
          'энтропия частот символов:', fixed['marginal_entropy'])


if __name__ == '__main__':
    main()
