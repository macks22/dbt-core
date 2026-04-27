"""Hash manifest.json after stripping known non-deterministic content.

Two clean parses of the same project on dbt 1.11.6 do not produce a
byte-identical ``manifest.json`` even with ``--no-partial-parse``. The
known sources of run-to-run variation are:

* Timestamps captured at parse time (``generated_at``, ``created_at``,
  ``compiled_at``, ``invocation_id``, plus values rendered by macros that
  call ``dbt.utils.current_timestamp``).
* Order of multi-source ``depends_on.nodes`` and ``sources`` lists for
  some models — appears to come from parser threading, not the model SQL.

This normalizer canonicalizes those so the hash reflects only semantic
content. Used as a regression check: after a refactor (e.g. the lazy
macro namespace), the normalized hash for each project should match its
baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# Top-level / per-node fields that reflect when the parse ran.
TIMESTAMP_KEYS = frozenset({
    "generated_at",
    "invocation_id",
    "user_id",
    "created_at",
    "compiled_at",
    "run_started_at",
    "process_user_id",
    "invocation_started_at",
    "send_anonymous_usage_stats",
    "dbt_version",
    "dbt_schema_version",
    "project_id",
})

# Lists where dbt does not guarantee a stable order across runs. Sorting
# preserves semantic content (membership) while erasing the spurious diff.
SORTED_LIST_KEYS = frozenset({
    "nodes",  # depends_on.nodes
    "macros",  # depends_on.macros
    "sources",  # node.sources (List[List[str]])
    "refs",  # node.refs
    "metrics",  # node.metrics
})

# Macros like ``current_timestamp`` and ``datetime.now`` render the
# wall-clock time into config strings (e.g. ``incremental_predicates``,
# ``meta.dest``). Strip those to get a stable comparison.
# - ISO timestamps: ``2026-04-27 15:31:36.634046+00:00``
# - Compact date + slash suffix: ``20260427/112`` (date + hour + quarter)
_TS_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?"),
    re.compile(r"\b\d{8}/\d+"),
]


def _scrub_timestamps(value: str) -> str:
    for pat in _TS_PATTERNS:
        value = pat.sub("<TS>", value)
    return value


def _sort_key(value: Any) -> Any:
    # Lists nested in sortable lists (e.g. node.sources is List[List[str]])
    # need a hashable key; tuple-ify them.
    if isinstance(value, list):
        return tuple(_sort_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _sort_key(v)) for k, v in value.items()))
    return value


def _normalize(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v, str(k)) for k, v in sorted(value.items()) if k not in TIMESTAMP_KEYS}
    if isinstance(value, list):
        normalized = [_normalize(v) for v in value]
        if parent_key in SORTED_LIST_KEYS:
            normalized.sort(key=_sort_key)
        return normalized
    if isinstance(value, str):
        return _scrub_timestamps(value)
    return value


def normalized_hash(path: Path) -> str:
    with path.open() as f:
        data = json.load(f)
    normalized = _normalize(data)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(normalized_hash(args.manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
