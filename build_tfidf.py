#!/usr/bin/env python3
"""
Подсчёт TF-IDF для терминов и лемм по скачанным документам (Задание 1) и спискам из Задания 2.

Для каждого документа:
- TF термина = число вхождений термина / общее число терминов в документе
- IDF термина = log(N / df), где N — число документов, df — число документов, содержащих термин
- TF леммы = (сумма вхождений всех терминов, приводящихся к лемме) / общее число терминов
- IDF леммы = log(N / df)

Выход:
- tfidf_terms/<N>.txt — по одному файлу на документ: <термин> <idf> <tf-idf>
- tfidf_lemmas/<N>.txt — по одному файлу на документ: <лемма> <idf> <tf-idf>
"""

import os
import re
import math
from collections import Counter

import pymorphy2
import pymorphy2_dicts_ru

from lemma_token_builder import process_page_with_counts

PAGES_DIR = "pages"
OUTPUT_TERMS_DIR = "tfidf_terms"
OUTPUT_LEMMAS_DIR = "tfidf_lemmas"

PAGE_NUM_PATTERN = re.compile(r"^(\d+)\.txt$")


def page_number_from_path(path: str) -> int | None:
    """Из пути pages/N.txt извлекает N."""
    name = os.path.basename(path)
    m = PAGE_NUM_PATTERN.match(name)
    return int(m.group(1)) if m else None


def process_document(html_path: str, morph, bad_pos: set) -> tuple[Counter[str], dict[str, int], int] | None:
    """
    Обрабатывает один документ. Возвращает (term_counts, lemma_counts, total_terms) или None при ошибке.
    """
    if not os.path.isfile(html_path):
        return None
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    # tokens_list: все термины документа с повторами (нужны для TF)
    # lemma_to_tokens: лемма -> набор словоформ в этом документе
    tokens_list, lemma_to_tokens = process_page_with_counts(html, morph, bad_pos)
    total_terms = len(tokens_list)

    # term_counts[t] = сколько раз термин t встретился в документе
    term_counts = Counter(tokens_list)

    # lemma_counts[l] = суммарная частота всех словоформ леммы l в документе:
    # sum(term_counts[форма]) по всем формам этой леммы
    lemma_counts: dict[str, int] = {}
    for lemma, tokens in lemma_to_tokens.items():
        lemma_counts[lemma] = sum(term_counts.get(t, 0) for t in tokens)
    return term_counts, lemma_counts, total_terms


def build_tfidf():
    if not os.path.isdir(PAGES_DIR):
        raise FileNotFoundError(
            f"Папка {PAGES_DIR}/ не найдена. Сначала запустите краулер (crawler.py)."
        )

    dict_path = pymorphy2_dicts_ru.get_path()
    morph = pymorphy2.MorphAnalyzer(path=dict_path)
    bad_pos = {"PREP", "CONJ", "PRCL", "INTJ"}

    # Собираем все страницы
    page_files = []
    for name in os.listdir(PAGES_DIR):
        if name.endswith(".txt"):
            num = page_number_from_path(os.path.join(PAGES_DIR, name))
            if num is not None:
                page_files.append((num, os.path.join(PAGES_DIR, name)))

    page_files.sort(key=lambda x: x[0])
    if not page_files:
        raise FileNotFoundError(f"В {PAGES_DIR}/ нет подходящих .txt файлов.")

    # Проход 1:
    # - считаем частоты терминов/лемм внутри каждого документа
    # - одновременно считаем document frequency (df):
    #   в скольких документах встретился термин/лемма
    doc_data: list[tuple[int, Counter[str], dict[str, int], int]] = []
    term_doc_freq: Counter[str] = Counter()
    lemma_doc_freq: Counter[str] = Counter()

    for doc_id, html_path in page_files:
        result = process_document(html_path, morph, bad_pos)
        if result is None:
            continue
        term_counts, lemma_counts, total_terms = result
        if total_terms == 0:
            continue
        doc_data.append((doc_id, term_counts, lemma_counts, total_terms))
        for term in term_counts:
            term_doc_freq[term] += 1
        for lemma in lemma_counts:
            lemma_doc_freq[lemma] += 1

    N = len(doc_data)
    if N == 0:
        raise ValueError("Нет документов с ненулевым числом терминов.")

    # IDF по условию задания:
    # idf = log(N / df), где
    # N  - число документов,
    # df - число документов, содержащих термин/лемму.
    # max(df, 1) оставлен как защитный механизм от деления на 0.
    def idf_term(term: str) -> float:
        df = term_doc_freq.get(term, 0)
        return math.log(N / max(df, 1))

    def idf_lemma(lemma: str) -> float:
        df = lemma_doc_freq.get(lemma, 0)
        return math.log(N / max(df, 1))

    os.makedirs(OUTPUT_TERMS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LEMMAS_DIR, exist_ok=True)

    for doc_id, term_counts, lemma_counts, total_terms in doc_data:
        # Файл по терминам: <термин> <idf> <tf-idf>
        terms_path = os.path.join(OUTPUT_TERMS_DIR, f"{doc_id}.txt")
        with open(terms_path, "w", encoding="utf-8") as f:
            for term in sorted(term_counts.keys()):
                # TF термина: сколько раз термин встретился в документе,
                # делим на общее число терминов в документе.
                tf = term_counts[term] / total_terms
                idf = idf_term(term)
                tfidf = tf * idf
                f.write(f"{term} {idf} {tfidf}\n")

        # Файл по леммам: <лемма> <idf> <tf-idf>
        lemmas_path = os.path.join(OUTPUT_LEMMAS_DIR, f"{doc_id}.txt")
        with open(lemmas_path, "w", encoding="utf-8") as f:
            for lemma in sorted(lemma_counts.keys()):
                # TF леммы: по условию задания это отношение
                # суммы вхождений всех терминов этой леммы
                # к общему числу терминов в документе.
                tf = lemma_counts[lemma] / total_terms
                idf = idf_lemma(lemma)
                tfidf = tf * idf
                f.write(f"{lemma} {idf} {tfidf}\n")

    print(f"TF-IDF посчитан для {N} документов.")
    print(f"Термины: {OUTPUT_TERMS_DIR}/")
    print(f"Леммы:   {OUTPUT_LEMMAS_DIR}/")


if __name__ == "__main__":
    build_tfidf()
