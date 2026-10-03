import json
from collections import defaultdict, Counter
import re

with open('/Users/ray/Documents/Fork/DeepLearning/class4/HW_train_data.json', 'r') as f:
    data = json.load(f)

# Text length distribution
lengths = sorted([len(s['text']) for s in data])
n = len(lengths)
print(f'Total samples: {n}')
print(f'Text length: min={lengths[0]}, max={lengths[-1]}, mean={sum(lengths)/n:.1f}, median={lengths[n//2]}')

e_counts = [len(s['entities']) for s in data]
r_counts = [len(s['relations']) for s in data]
print(f'Entities/sample: min={min(e_counts)}, max={max(e_counts)}, mean={sum(e_counts)/n:.1f}, median={sorted(e_counts)[n//2]}')
print(f'Relations/sample: min={min(r_counts)}, max={max(r_counts)}, mean={sum(r_counts)/n:.1f}, median={sorted(r_counts)[n//2]}')

# Overlap check
overlap_details = defaultdict(list)
for s in data:
    spans = {}
    for e in s['entities']:
        key = (e['start'], e['end'])
        if key in spans:
            overlap_details[spans[key]].append((e['label'], e['text'], s['text'][:60]))
        spans[key] = e['label']

print(f'\nOverlapping entities (same span, diff label):')
for k, v in overlap_details.items():
    print(f'  {k} overlapped with:')
    for other_label, text, ctx in v[:3]:
        print(f'    -> {other_label}: "{text}" ...{ctx}')

# Language check
has_chinese = sum(1 for s in data if re.search(r'[\u4e00-\u9fff]', s['text']))
has_english = sum(1 for s in data if re.search(r'[a-zA-Z]', s['text']))
print(f'\nSamples with Chinese: {has_chinese}, with English: {has_english}')

# Relation entity match check
mismatch = 0
for s in data:
    entity_spans = {(e['start'], e['end']): e for e in s['entities']}
    for r in s['relations']:
        h_key = (r['head_start'], r['head_end'])
        t_key = (r['tail_start'], r['tail_end'])
        if h_key not in entity_spans or t_key not in entity_spans:
            mismatch += 1

print(f'Relation head/tail not in entities: {mismatch}')

# Entity and relation labels
all_ent_labels = set()
all_rel_labels = set()
for s in data:
    for e in s['entities']:
        all_ent_labels.add(e['label'])
    for r in s['relations']:
        all_rel_labels.add(r['label'])
print(f'Entity labels: {sorted(all_ent_labels)}')
print(f'Relation labels: {sorted(all_rel_labels)}')

# Print a few Chinese samples
print('\n=== Chinese Samples ===')
cn_count = 0
for s in data:
    if re.search(r'[\u4e00-\u9fff]', s['text']):
        print(f'\nText: {s["text"]}')
        for e in s['entities']:
            print(f'  [{e["label"]}] "{e["text"]}" ({e["start"]}-{e["end"]})')
        for r in s['relations']:
            print(f'  [{r["label"]}] "{r["head"]}" -> "{r["tail"]}"')
        cn_count += 1
        if cn_count >= 2:
            break

# Check how many samples have zero entities or relations
zero_ent = sum(1 for s in data if len(s['entities']) == 0)
zero_rel = sum(1 for s in data if len(s['relations']) == 0)
print(f'\nSamples with 0 entities: {zero_ent}, with 0 relations: {zero_rel}')