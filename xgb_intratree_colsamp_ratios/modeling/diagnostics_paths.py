from __future__ import annotations

import json
from dataclasses import dataclass

from xgboost import XGBClassifier


@dataclass(frozen=True)
class CooccurrenceResult:
    per_pair: dict[str, float]
    mean: float


@dataclass(frozen=True)
class PathFractionResult:
    per_pair: dict[str, float]
    mean: float


def _required_feature_masks(features: list[str]) -> tuple[dict[str, int], int]:
    idx = {f: i for i, f in enumerate(features)}
    full = (1 << len(features)) - 1 if features else 0
    return idx, full


def _tree_has_all_features(node: dict, idx: dict[str, int], full_mask: int, seen_mask: int) -> bool:
    if "leaf" in node:
        return bool((seen_mask & full_mask) == full_mask)

    split = node.get("split")
    if split in idx:
        seen_mask |= 1 << idx[split]
        if (seen_mask & full_mask) == full_mask:
            return True

    children = node.get("children") or []
    for child in children:
        if _tree_has_all_features(child, idx, full_mask, seen_mask):
            return True
    return False


def _tree_has_pair(node: dict, f1: str, f2: str, seen1: bool, seen2: bool) -> bool:
    if "leaf" in node:
        return bool(seen1 and seen2)

    split = node.get("split")
    if split == f1:
        seen1 = True
    elif split == f2:
        seen2 = True
    if seen1 and seen2:
        return True

    children = node.get("children") or []
    for child in children:
        if _tree_has_pair(child, f1, f2, seen1, seen2):
            return True
    return False


def tree_level_pair_cooccurrence_fraction(model: XGBClassifier, f1: str, f2: str) -> float:
    booster = model.get_booster()
    trees = booster.get_dump(dump_format="json")
    if not trees:
        return 0.0
    hits = 0
    for tree_str in trees:
        root = json.loads(tree_str)
        if _tree_has_pair(root, f1, f2, False, False):
            hits += 1
    return float(hits) / float(len(trees))


def cooccurrence_for_pairs(model: XGBClassifier, pairs: list[tuple[str, str]]) -> CooccurrenceResult:
    per_pair: dict[str, float] = {}
    vals = []
    for f1, f2 in pairs:
        key = f"{f1}__{f2}"
        val = tree_level_pair_cooccurrence_fraction(model, f1, f2)
        per_pair[key] = val
        vals.append(val)
    mean = float(sum(vals) / len(vals)) if vals else 0.0
    return CooccurrenceResult(per_pair=per_pair, mean=mean)


def tree_level_all_features_cooccurrence_fraction(model: XGBClassifier, features: list[str]) -> float:
    idx, full_mask = _required_feature_masks(features)
    if full_mask == 0:
        return 0.0
    booster = model.get_booster()
    trees = booster.get_dump(dump_format="json")
    if not trees:
        return 0.0
    hits = 0
    for tree_str in trees:
        root = json.loads(tree_str)
        if _tree_has_all_features(root, idx, full_mask, 0):
            hits += 1
    return float(hits) / float(len(trees))


def _leaf_path_weight(node: dict) -> float:
    w = node.get("cover", None)
    if w is None:
        return 1.0
    try:
        w_f = float(w)
    except (TypeError, ValueError):
        return 1.0
    if not (w_f > 0.0):
        return 1.0
    return w_f


def _tree_leaf_path_fraction(
    node: dict,
    f1: str,
    f2: str,
    seen1: bool,
    seen2: bool,
) -> tuple[float, float]:
    if "leaf" in node:
        w = _leaf_path_weight(node)
        hit = 1.0 if (seen1 and seen2) else 0.0
        return hit * w, w

    split = node.get("split")
    if split == f1:
        seen1 = True
    elif split == f2:
        seen2 = True

    hit_w = 0.0
    tot_w = 0.0
    children = node.get("children") or []
    for child in children:
        h, t = _tree_leaf_path_fraction(child, f1, f2, seen1, seen2)
        hit_w += h
        tot_w += t
    return hit_w, tot_w


def tree_level_pair_path_fraction(model: XGBClassifier, f1: str, f2: str) -> float:
    booster = model.get_booster()
    trees = booster.get_dump(dump_format="json")
    if not trees:
        return 0.0

    vals: list[float] = []
    for tree_str in trees:
        root = json.loads(tree_str)
        hit_w, tot_w = _tree_leaf_path_fraction(root, f1, f2, False, False)
        vals.append(float(hit_w / tot_w) if tot_w > 0 else 0.0)
    return float(sum(vals) / len(vals)) if vals else 0.0


def path_fraction_for_pairs(model: XGBClassifier, pairs: list[tuple[str, str]]) -> PathFractionResult:
    per_pair: dict[str, float] = {}
    vals = []
    for f1, f2 in pairs:
        key = f"{f1}__{f2}"
        val = tree_level_pair_path_fraction(model, f1, f2)
        per_pair[key] = val
        vals.append(val)
    mean = float(sum(vals) / len(vals)) if vals else 0.0
    return PathFractionResult(per_pair=per_pair, mean=mean)


def _tree_leaf_all_features_path_fraction(
    node: dict,
    idx: dict[str, int],
    full_mask: int,
    seen_mask: int,
) -> tuple[float, float]:
    if "leaf" in node:
        w = _leaf_path_weight(node)
        hit = 1.0 if ((seen_mask & full_mask) == full_mask) else 0.0
        return hit * w, w

    split = node.get("split")
    if split in idx:
        seen_mask |= 1 << idx[split]

    hit_w = 0.0
    tot_w = 0.0
    children = node.get("children") or []
    for child in children:
        h, t = _tree_leaf_all_features_path_fraction(child, idx, full_mask, seen_mask)
        hit_w += h
        tot_w += t
    return hit_w, tot_w


def tree_level_all_features_path_fraction(model: XGBClassifier, features: list[str]) -> float:
    idx, full_mask = _required_feature_masks(features)
    if full_mask == 0:
        return 0.0
    booster = model.get_booster()
    trees = booster.get_dump(dump_format="json")
    if not trees:
        return 0.0
    vals: list[float] = []
    for tree_str in trees:
        root = json.loads(tree_str)
        hit_w, tot_w = _tree_leaf_all_features_path_fraction(root, idx, full_mask, 0)
        vals.append(float(hit_w / tot_w) if tot_w > 0 else 0.0)
    return float(sum(vals) / len(vals)) if vals else 0.0
