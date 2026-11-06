import time
import numpy as np

_DEADLINE = None
_START = None

def _now():
    return time.perf_counter()

def _time_left():
    if _DEADLINE is None:
        return float('inf')
    return max(0.0, _DEADLINE - _now())

def _validity_check(triples):
    """Return list of violating 6-point sets (each a frozenset)."""
    if len(triples) < 8:
        return []
    arr = np.array(triples, dtype=np.int64)
    if arr.size == 0:
        return []
    n = 16
    # For each of the 1050 6-sets, count how many of its 20 triples are present.
    # Build triple->index map.
    key = (arr[:, 0] * 256 + arr[:, 1] * 16 + arr[:, 2])
    present = {}
    for i in range(len(arr)):
        present[int(key[i])] = i
    viol = []
    # iterate all 6-subsets
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    for e in range(d + 1, n):
                        for f in range(e + 1, n):
                            S = (a, b, c, d, e, f)
                            # 20 triples inside S
                            cnt = 0
                            # enumerate all 20 index-triples of S
                            idx = list(range(6))
                            for i in range(6):
                                for j in range(i + 1, 6):
                                    for k in range(j + 1, 6):
                                        x, y, z = S[idx[i]], S[idx[j]], S[idx[k]]
                                        kk = x * 256 + y * 16 + z
                                        if kk in present:
                                            cnt += 1
                                            if cnt >= 8:
                                                break
                                    if cnt >= 8:
                                        break
                                if cnt >= 8:
                                    break
                            if cnt >= 8:
                                viol.append(frozenset(S))
    return viol

def _greedy_remove(triples, viol):
    """Remove fewest triples to hit all violations (greedy set cover)."""
    if not viol:
        return set()
    # For each violation, the set of triples contained in it.
    tri_set = set(map(tuple, triples))
    # map triple -> list of violation indices it belongs to
    from collections import defaultdict
    tri_to_viol = defaultdict(list)
    for vi, S in enumerate(viol):
        Slist = sorted(S)
        for i in range(6):
            for j in range(i + 1, 6):
                for k in range(j + 1, 6):
                    t = (Slist[i], Slist[j], Slist[k])
                    if t in tri_set:
                        tri_to_viol[t].append(vi)
    remaining = [set(v) for v in tri_to_viol.values()]
    # we need to cover all violation indices 0..len(viol)-1
    # each triple covers the set of violations it's in
    cover_sets = {t: set(vs) for t, vs in tri_to_viol.items()}
    uncovered = set(range(len(viol)))
    removed = set()
    while uncovered:
        # pick triple covering most uncovered
        best = None
        best_cnt = 0
        for t, vs in cover_sets.items():
            if t in removed:
                continue
            c = len(vs & uncovered)
            if c > best_cnt:
                best_cnt = c
                best = t
        if best is None:
            break
        removed.add(best)
        uncovered -= cover_sets[best]
    return removed

def _add_back_pass(triples, candidates, deadline):
    """Try to re-add candidate triples that don't create a violation."""
    cur = list(triples)
    cur_set = set(map(tuple, cur))
    for t in candidates:
        if _time_left() < 3.0:
            break
        if t in cur_set:
            continue
        trial = cur + [list(t)]
        v = _validity_check(trial)
        if not v:
            cur.append(list(t))
            cur_set.add(t)
    return cur

def construct():
    global _DEADLINE, _START
    if _DEADLINE is None:
        _START = _now()
        _DEADLINE = _START + 135.0  # safety margin under 150s
    else:
        _START = _now()
    n = 16
    # All 560 triples
    all_tris = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                all_tris.append((a, b, c))
    # pair -> list of triples containing it
    pair_to_tris = {}
    for t in all_tris:
        for i in range(3):
            for j in range(i + 1, 3):
                p = (t[i], t[j])
                pair_to_tris.setdefault(p, []).append(t)
    # pair load
    load = {p: 0 for p in pair_to_tris}
    selected = []
    sel_set = set()
    # greedy: always pick triple minimizing max pair-load among its pairs
    import heapq
    # We'll recompute best each iteration but that's O(560*3) which is fine.
    remaining = list(all_tris)
    while remaining and _time_left() > 6.0:
        best = None
        best_key = None
        for t in remaining:
            m = max(load[(t[0], t[1])], load[(t[1], t[2])], load[(t[0], t[2])])
            if best_key is None or m < best_key:
                best_key = m
                best = t
        if best is None:
            break
        selected.append(best)
        sel_set.add(best)
        for i in range(3):
            for j in range(i + 1, 3):
                load[(best[i], best[j])] += 1
        remaining.remove(best)
    # Now we have a balanced set. Check validity.
    triples = [list(t) for t in selected]
    viol = _validity_check(triples)
    removed = set()
    if viol:
        removed = _greedy_remove(triples, viol)
        triples = [t for t in triples if tuple(t) not in removed]
        # verify
        viol2 = _validity_check(triples)
        if viol2:
            # fallback: keep removing until valid
            for t in list(triples):
                if not viol2:
                    break
                trial = [x for x in triples if x != t]
                if not _validity_check(trial):
                    triples = trial
                    viol2 = _validity_check(triples)
    # add-back pass with remaining (unselected) triples
    candidates = [t for t in all_tris if t not in sel_set]
    triples = _add_back_pass(triples, candidates, _DEADLINE)
    # final validity check
    if _validity_check(triples):
        # emergency repair
        removed2 = _greedy_remove(triples, _validity_check(triples))
        triples = [t for t in triples if tuple(t) not in removed2]
    return triples

if __name__ == "__main__":
    t = construct()
    print(len(t))
