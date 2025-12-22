# io_utils.py
import json
import os
import operator

def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

def filter(results: list, field: str, op: str, expected) -> list:
    func = OPS[op]
    out = []
    for r in results:
        # score might be top-level; success is usually inside result
        value = r.get(field, r.get("result", {}).get(field))
        try:
            if func(value, expected):
                out.append(r)
        except Exception:
            continue
    return out
