import threading
import time
import json
import os

HAVUZ_PATH = os.path.join(os.path.dirname(__file__), 'result_havuzu.json')
_LOCK = threading.Lock()

def _load():
    if not os.path.exists(HAVUZ_PATH):
        return []
    with open(HAVUZ_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return []

def _save(results):
    with open(HAVUZ_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

def add_result(result):
    with _LOCK:
        results = _load()
        results.append(result)
        _save(results)

def get_results_and_clear(older_than_seconds=60):
    now = time.time()
    with _LOCK:
        results = _load()
        recent = [r for r in results if now - r['timestamp'] <= older_than_seconds]
        kalan = [r for r in results if now - r['timestamp'] > older_than_seconds]
        _save(kalan)
    return recent
