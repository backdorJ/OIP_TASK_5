#!/usr/bin/env python3
"""
Булев поиск по инвертированному индексу.

Поддерживаются операторы AND, OR, NOT и скобки для сложных запросов.
Запрос вводится строкой (например: (Клеопатра AND Цезарь) OR (Антоний AND Цицерон) OR Помпей).

Использование:
  python boolean_search.py
  затем ввести запрос с клавиатуры.

Или передать запрос аргументом:
  python boolean_search.py "(москва AND россия) OR санкт"
"""

import os
import re
import sys

INVERTED_INDEX_FILE = "inverted_index.txt"
INDEX_TXT = "index.txt"  # doc_id -> URL

# Операторы в запросе (регистронезависимо)
AND_OP = "and"
OR_OP = "or"
NOT_OP = "not"

def load_inverted_index(path=INVERTED_INDEX_FILE):
    """Загружает инвертированный индекс: терм -> множество doc_id."""
    index = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл {path} не найден. Сначала запустите build_inverted_index.py")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            term = parts[0].lower()
            doc_ids = set(int(x) for x in parts[1:])
            index[term] = doc_ids
    return index


def load_doc_urls(path=INDEX_TXT):
    """Загружает соответствие doc_id -> URL из index.txt."""
    urls = {}
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


def tokenize_query(s):
    """
    Разбивает строку запроса на токены: слова (термы), AND, OR, NOT, (, ).
    Термы приводятся к нижнему регистру.
    """
    s = s.strip()
    if not s:
        return []
    # Разделяем по пробелам и скобкам, сохраняя скобки отдельно
    tokens = re.findall(r'[()]|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', s, re.IGNORECASE)
    result = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        upper = t.upper()
        if upper == "AND":
            result.append(AND_OP)
        elif upper == "OR":
            result.append(OR_OP)
        elif upper == "NOT":
            result.append(NOT_OP)
        elif t in "()":
            result.append(t)
        else:
            result.append(t.lower())
    return result

class QueryParser:
    """Рекурсивный разбор булева выражения. Приоритет: OR < AND < NOT < терм/скобки."""

    def __init__(self, tokens, index, all_doc_ids):
        self.tokens = tokens
        self.pos = 0
        self.index = index
        self.all_doc_ids = all_doc_ids

    def peek(self):
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def consume(self):
        if self.pos < len(self.tokens):
            self.pos += 1

    def term_set(self, term):
        """Возвращает множество doc_id для терма (леммы)."""
        return self.index.get(term, set()).copy()

    def parse_primary(self):
        """primary = терм | ( expression ) | NOT primary"""
        t = self.peek()
        if t is None:
            return set()
        if t == "(":
            self.consume()
            s = self.parse_or()
            if self.peek() == ")":
                self.consume()
            return s
        if t == NOT_OP:
            self.consume()
            inner = self.parse_primary()
            return self.all_doc_ids - inner
        if t not in (AND_OP, OR_OP, ")", None):
            term = t
            self.consume()
            return self.term_set(term)
        return set()

    def parse_and(self):
        left = self.parse_primary()
        while self.peek() == AND_OP:
            self.consume()
            right = self.parse_primary()
            left = left & right
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == OR_OP:
            self.consume()
            right = self.parse_and()
            left = left | right
        return left

    def parse(self):
        return self.parse_or()

def boolean_search(query_str, index, all_doc_ids):
    """Выполняет булев поиск по строке запроса. Возвращает множество doc_id."""
    tokens = tokenize_query(query_str)
    if not tokens:
        return set()
    parser = QueryParser(tokens, index, all_doc_ids)
    return parser.parse()

def main():
    index = load_inverted_index()
    doc_urls = load_doc_urls()
    all_doc_ids = set(doc_urls.keys()) if doc_urls else set()
    # Если index.txt пустой/отсутствует — все doc_id из индекса
    if not all_doc_ids:
        for doc_set in index.values():
            all_doc_ids |= doc_set

    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
    else:
        print("Булев поиск. Операторы: AND, OR, NOT. Скобки для группировки.")
        print("Пример: (Клеопатра AND Цезарь) OR (Антоний AND Цицерон) OR Помпей")
        print()
        try:
            query_str = input("Введите запрос: ").strip()
        except EOFError:
            query_str = ""

    if not query_str:
        print("Запрос пустой.")
        return

    doc_ids = boolean_search(query_str, index, all_doc_ids)
    doc_ids_sorted = sorted(doc_ids)

    print(f"\nНайдено документов: {len(doc_ids_sorted)}")
    for doc_id in doc_ids_sorted:
        url = doc_urls.get(doc_id, "")
        print(f"  {doc_id}\t{url}")


if __name__ == "__main__":
    main()
