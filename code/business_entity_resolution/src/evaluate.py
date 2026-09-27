"""F_beta (macro over Source-1 entities) scoring, matching the challenge's
evaluation formula exactly, including the singleton convention: a S1 entity
with no true matches scores 1.0 when predicted empty, 0.0 if any match is
predicted for it."""


def f_beta_score(pred: set, truth: set, beta: float = 0.5) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def f_beta_macro(pred_map: dict, truth_map: dict, beta: float = 0.5) -> float:
    """``pred_map`` / ``truth_map``: source1_entity_id -> set of matched ids.
    Averaged over every key in ``truth_map`` (missing keys in ``pred_map``
    are treated as an empty prediction)."""
    scores = [
        f_beta_score(pred_map.get(s1, set()), truth, beta=beta)
        for s1, truth in truth_map.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0


def parse_id_list_column(series) -> dict:
    """``series`` is a Polars (source1_entity_id, id_list_str) frame's rows as
    tuples; returns source1_entity_id -> set(ids), empty set for ""."""
    out = {}
    for s1, ids in series:
        out[s1] = set(ids.split(",")) if ids else set()
    return out
