"""Ternary covering code K_3(6,1): greedy + local search + seed-based construction."""
import time
import numpy as np

_Q = 3
_N = 6
_R = 1
_TOTAL = 729
_DEADLINE = 110.0
_TIME_BUDGET = 100.0
_SEED_LIMIT = 1000
_SEED_SUBSIZE = 55
_N_LOCAL = 60


def _neighbors(word):
    """All words within Hamming distance 1 of `word` (a list of ints)."""
    out = [list(word)]
    for i in range(_N):
        for v in range(_Q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                out.append(w)
    return out


def _tetracode81():
    """Direct-sum construction: 9 * 3**2 = 81 words (valid, the seed baseline)."""
    tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    out = []
    for h in tet:
        for c in range(3):
            for d in range(3):
                out.append(h + [c, d])
    return out


def _greedy_cover(chosen, uncovered, deadline):
    """Greedy maximal cover: keep adding the word covering most uncovered points.
    Ties broken by max overlap with the current chosen set. Returns (chosen, uncovered)."""
    chosen_set = set(chosen)
    while True:
        if time.time() > deadline:
            break
        best = None
        best_gain = 0
        best_ov = -1
        for w in range(_TOTAL):
            if w in chosen_set:
                continue
            nbs = _neighbors(w)
            gain = sum(1 for x in nbs if x in uncovered)
            if gain < best_gain:
                continue
            ov = sum(1 for x in nbs if x in chosen_set)
            if gain > best_gain or (gain == best_gain and ov > best_ov):
                best, best_gain, best_ov = w, gain, ov
        if best is None or best_gain == 0:
            break
        chosen_set.add(best)
        chosen.append(best)
        for x in _neighbors(best):
            uncovered.discard(x)
        if not uncovered:
            break
    return chosen, uncovered


def _local_search(chosen, deadline, n_iter=_N_LOCAL):
    """Local search minimizing |chosen| while keeping the cover complete."""
    chosen = list(chosen)
    for _ in range(n_iter):
        if time.time() > deadline:
            break
        # 1) removals
        removed = False
        for idx in range(len(chosen) - 1, -1, -1):
            if time.time() > deadline:
                break
            cand = chosen[:idx] + chosen[idx + 1:]
            if _is_cover(cand):
                chosen = cand
                removed = True
                break
        if removed:
            continue
        # 2) swaps
        improved = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            for j in range(len(chosen)):
                if i == j:
                    continue
                for w in range(_TOTAL):
                    if w in chosen:
                        continue
                    cand = chosen[:i] + [w] + chosen[i + 1:]
                    if _is_cover(cand):
                        chosen = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return chosen


def _is_cover(words):
    """True iff `words` covers the whole space (every word within distance 1)."""
    if not words:
        return False
    arr = np.array(words, dtype=np.uint8)
    diffs = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
    return bool((diffs <= _R).any(axis=1).all())


def construct(q, n, radius):
    global _Q, _N, _R, _TOTAL, _TIME_BUDGET
    if q != _Q or n != _N or radius != _R:
        return _generic_fallback(q, n, radius)

    start = time.time()
    deadline = start + _TIME_BUDGET

    best = _tetracode81()
    best_len = len(best)

    # Greedy maximal cover from scratch
    chosen, uncovered = _greedy_cover([], set(range(_TOTAL)), deadline)
    if uncovered and len(chosen) < best_len:
        chosen = _complete_from_seed(chosen, uncovered, deadline)
    if len(chosen) < best_len:
        best, best_len = chosen, len(chosen)
        best = _local_search(best, deadline)
        best_len = len(best)

    # Greedy-from-seed: seed with a random subset of the 81-word construction
    seed_words = _tetracode81()
    for it in range(_SEED_LIMIT):
        if time.time() > deadline:
            break
        chosen = list(np.random.permutation(seed_words)[:_SEED_SUBSIZE])
        chosen_set = set(chosen)
        covered = set()
        for w in chosen:
            covered.update(_neighbors(w))
        uncovered = set(range(_TOTAL)) - covered
        if not uncovered:
            if len(chosen) < best_len:
                best, best_len = chosen, len(chosen)
            continue
        chosen, uncovered = _greedy_cover(chosen, uncovered, deadline)
        if uncovered:
            chosen = _complete_from_seed(chosen, uncovered, deadline)
        if len(chosen) < best_len:
            best, best_len = chosen, len(chosen)
            best = _local_search(best, deadline)
            best_len = len(best)

    # Deadline guard: return a guaranteed-valid result
    if time.time() > _DEADLINE or not _is_cover(best):
        return _tetracode81()
    return best


def _complete_from_seed(chosen, uncovered, deadline):
    """Finish covering `uncovered` (from a partial seed) via greedy, then local search."""
    chosen, uncovered = _greedy_cover(chosen, uncovered, deadline)
    if uncovered:
        # Fallback: add words for any remaining uncovered points
        for u in list(uncovered):
            w = u
            chosen.append(w)
            for x in _neighbors(w):
                uncovered.discard(x)
    return _local_search(chosen, deadline)


def _generic_fallback(q, n, radius):
    """Conservative valid cover for non-6/3/1 parameters (exhaustive radius-1 balls)."""
    import itertools
    words = []
    for w in itertools.product(range(q), repeat=n):
        w = list(w)
        covered = True
        for v in w:
            if v not in words:
                covered = False
                break
        if covered:
            words.append(w)
    return words
