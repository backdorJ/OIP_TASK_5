#!/usr/bin/env python3
"""
Векторный поиск по корпусу документов на основе TF-IDF по леммам.

Использует уже посчитанные файлы из папки tfidf_lemmas/ (build_tfidf.py)
и ранжирует документы по косинусному сходству с запросом.

Формат файлов tfidf_lemmas/<N>.txt:
<лемма> <idf> <tf-idf>

Запуск:
  python vector_search.py "пример поискового запроса"
или без аргументов — запрос будет запрошен с клавиатуры.
"""

import inspect
if not hasattr(inspect, "getargspec"):
    def _getargspec(f):
        spec = inspect.getfullargspec(f)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = _getargspec

import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pymorphy2
import pymorphy2_dicts_ru

TFIDF_LEMMAS_DIR = "tfidf_lemmas"
INDEX_TXT = "index.txt"  # doc_id -> URL

# Только русские слова (включая ё), длина >= 2
RE_RU_WORD = re.compile(r"[а-яё]{2,}", re.IGNORECASE)

BAD_POS = {"PREP", "CONJ", "PRCL", "INTJ"}


def load_doc_urls(path: str = INDEX_TXT) -> Dict[int, str]:
    """Загружает соответствие doc_id -> URL из index.txt."""
    urls: Dict[int, str] = {}
    if not os.path.isfile(path):
        return urls
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) >= 2:
                doc_id = int(parts[0])
                urls[doc_id] = parts[1].strip()
            elif len(parts) == 1 and parts[0].isdigit():
                urls[int(parts[0])] = ""
    return urls


def load_tfidf_lemmas(
    directory: str = TFIDF_LEMMAS_DIR,
) -> Tuple[Dict[int, Dict[str, float]], Counter, int]:
    """
    Загружает TF-IDF по леммам из tfidf_lemmas/<doc_id>.txt.

    Возвращает:
      - doc_vectors: doc_id -> {lemma: tfidf}
      - lemma_df: количество документов, в которых встречается лемма
      - N: число документов
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Папка {directory}/ не найдена. Сначала запустите build_tfidf.py."
        )

    doc_vectors: Dict[int, Dict[str, float]] = {}
    lemma_df: Counter = Counter()

    for name in os.listdir(directory):
        if not name.endswith(".txt"):
            continue
        base = os.path.splitext(name)[0]
        if not base.isdigit():
            continue
        doc_id = int(base)
        path = os.path.join(directory, name)
        vec: Dict[str, float] = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 2)
                if len(parts) != 3:
                    continue
                lemma, _idf_str, tfidf_str = parts
                try:
                    tfidf = float(tfidf_str)
                except ValueError:
                    continue
                if tfidf == 0.0:
                    continue
                vec[lemma] = tfidf
                lemma_df[lemma] += 1

        if vec:
            doc_vectors[doc_id] = vec

    N = len(doc_vectors)
    if N == 0:
        raise ValueError(f"В {directory}/ не найдено ни одного ненулевого вектора.")

    return doc_vectors, lemma_df, N


def build_doc_norms_and_postings(
    doc_vectors: Dict[int, Dict[str, float]]
) -> Tuple[Dict[int, float], Dict[str, List[Tuple[int, float]]]]:
    """
    По готовым векторам строит:
      - нормы документов ||d_i||
      - "обратный" индекс: лемма -> список (doc_id, weight)
    Это позволяет считать скалярное произведение только по нужным документам.
    """
    doc_norms: Dict[int, float] = {}
    postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    for doc_id, vec in doc_vectors.items():
        sq_sum = 0.0
        for lemma, w in vec.items():
            sq_sum += w * w
            postings[lemma].append((doc_id, w))
        doc_norms[doc_id] = math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0

    return doc_norms, postings


def init_morph():
    dict_path = pymorphy2_dicts_ru.get_path()
    return pymorphy2.MorphAnalyzer(path=dict_path)


def analyze_query_lemmas(query: str, morph) -> List[str]:
    """
    Приводит строку запроса к списку лемм так же, как страницы:
    - русские слова
    - фильтрация по частям речи
    - нормальная форма (лемма)
    """
    text = query.lower()
    raw_tokens = RE_RU_WORD.findall(text)

    lemmas: List[str] = []
    for tok in raw_tokens:
        if any(ch.isdigit() for ch in tok):
            continue
        parses = morph.parse(tok)
        if not parses:
            continue
        p = parses[0]
        pos = p.tag.POS
        lemma = p.normal_form
        if pos in BAD_POS:
            continue
        if not RE_RU_WORD.fullmatch(lemma) or not (2 <= len(lemma) <= 40):
            continue
        lemmas.append(lemma)
    return lemmas


def build_query_vector(
    query: str,
    morph,
    lemma_df: Counter,
    N_docs: int,
) -> Dict[str, float]:
    """
    Строит TF-IDF вектор запроса по леммам.
    IDF берётся из корпуса: idf = log(N / df).
    """
    lemmas = analyze_query_lemmas(query, morph)
    if not lemmas:
        return {}

    counts = Counter(lemmas)
    total = len(lemmas)
    vec: Dict[str, float] = {}

    for lemma, cnt in counts.items():
        df = lemma_df.get(lemma, 0)
        if df <= 0:
            # Лемма не встречается ни в одном документе — она не влияет на ранжирование
            continue
        tf = cnt / total
        idf = math.log(N_docs / df)
        tfidf = tf * idf
        if tfidf != 0.0:
            vec[lemma] = tfidf

    return vec


def cosine_similarities(
    query_vec: Dict[str, float],
    doc_norms: Dict[int, float],
    postings: Dict[str, List[Tuple[int, float]]],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Считает косинусное сходство между вектором запроса и всеми документами.
    Возвращает отсортированный по убыванию список (doc_id, score) длиной до top_k.
    """
    if not query_vec:
        return []

    # ||q||
    sq_sum_q = sum(w * w for w in query_vec.values())
    if sq_sum_q <= 0.0:
        return []
    norm_q = math.sqrt(sq_sum_q)

    # Скалярное произведение только по документам, содержащим леммы из запроса
    scores: Dict[int, float] = defaultdict(float)
    for lemma, wq in query_vec.items():
        for doc_id, wd in postings.get(lemma, []):
            scores[doc_id] += wq * wd

    # Превращаем в косинусное сходство
    result: List[Tuple[int, float]] = []
    for doc_id, dot in scores.items():
        norm_d = doc_norms.get(doc_id, 0.0)
        if norm_d <= 0.0:
            continue
        sim = dot / (norm_d * norm_q)
        if sim > 0.0:
            result.append((doc_id, sim))

    # Сортируем по убыванию сходства
    result.sort(key=lambda x: x[1], reverse=True)
    if top_k > 0:
        result = result[:top_k]
    return result


def vector_search(
    query: str,
    doc_vectors: Dict[int, Dict[str, float]],
    lemma_df: Counter,
    N_docs: int,
    doc_norms: Dict[int, float],
    postings: Dict[str, List[Tuple[int, float]]],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Полный цикл: строим вектор запроса и считаем косинусное сходство.
    """
    morph = init_morph()
    q_vec = build_query_vector(query, morph, lemma_df, N_docs)
    return cosine_similarities(q_vec, doc_norms, postings, top_k=top_k)


def main():
    # Загружаем TF-IDF по леммам
    doc_vectors, lemma_df, N_docs = load_tfidf_lemmas(TFIDF_LEMMAS_DIR)
    doc_norms, postings = build_doc_norms_and_postings(doc_vectors)
    doc_urls = load_doc_urls(INDEX_TXT)

    # Читаем запрос
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
    else:
        print("Векторный поиск по TF-IDF (леммы).")
        print("Пример запроса: Клеопатра Цезарь Антоний")
        print()
        try:
            query_str = input("Введите запрос: ").strip()
        except EOFError:
            query_str = ""

    if not query_str:
        print("Запрос пустой.")
        return

    TOP_K = 10
    results = vector_search(
        query_str,
        doc_vectors=doc_vectors,
        lemma_df=lemma_df,
        N_docs=N_docs,
        doc_norms=doc_norms,
        postings=postings,
        top_k=TOP_K,
    )

    if not results:
        print("По вашему запросу не найдено релевантных документов.")
        return

    print(f"\nТоп-{len(results)} документов по косинусному сходству:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        url = doc_urls.get(doc_id, "")
        print(f"{rank:2d}. doc_id={doc_id:3d}  score={score:.4f}  {url}")


if __name__ == "__main__":
    main()

