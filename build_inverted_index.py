#!/usr/bin/env python3
"""
Построение инвертированного индекса по токенизированным страницам.

Читает tokenized_pages/pageN/lemmas.txt, собирает для каждой леммы список ID документов,
в которых она встречается. Сохраняет результат в inverted_index.txt.

Формат inverted_index.txt:
<лемма> <doc_id1> <doc_id2> ... <doc_idN>

где doc_id — номер страницы (1, 2, 3, ...).
"""

import os
import re
import glob

TOKENIZED_DIR = "tokenized_pages"
OUTPUT_FILE = "inverted_index.txt"

# Папки page1, page2, ... извлекаем номер
PAGE_DIR_PATTERN = re.compile(r"page(\d+)$")


def build_inverted_index():
    index = {}  # lemma -> set of doc_ids

    if not os.path.isdir(TOKENIZED_DIR):
        raise FileNotFoundError(f"Папка {TOKENIZED_DIR}/ не найдена. Сначала запустите lemma_token_builder.py")

    page_dirs = sorted(
        glob.glob(os.path.join(TOKENIZED_DIR, "page*")),
        key=lambda p: int(PAGE_DIR_PATTERN.search(p).group(1)) if PAGE_DIR_PATTERN.search(p) else 0,
    )

    for page_path in page_dirs:
        match = PAGE_DIR_PATTERN.search(page_path)
        if not match:
            continue
        doc_id = int(match.group(1))
        lemmas_file = os.path.join(page_path, "lemmas.txt")
        if not os.path.isfile(lemmas_file):
            continue

        with open(lemmas_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                lemma = parts[0].lower()
                if lemma not in index:
                    index[lemma] = set()
                index[lemma].add(doc_id)

    # Сортируем леммы и для каждой — отсортированный список doc_id
    sorted_terms = sorted(index.keys())
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for term in sorted_terms:
            doc_ids = sorted(index[term])
            out.write(term + " " + " ".join(str(d) for d in doc_ids) + "\n")

    print(f"Инвертированный индекс записан в {OUTPUT_FILE}")
    print(f"Уникальных терминов: {len(index)}")
    return index


if __name__ == "__main__":
    build_inverted_index()
