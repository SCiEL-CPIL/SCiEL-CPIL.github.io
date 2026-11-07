"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

_N = 12
_M = 40
_DEADLINE = 85.0  # safety margin before 90s kill

def _count_comparable(family):
    """Count unordered pairs where one set is a strict subset of the other."""
    masks = [0] * _M
    for i, s in enumerate(family):
        m = 0
        for x in s:
            m |= (1 << x)
        masks[i] = m
    cnt = 0
    for i in range(_M):
        mi = masks[i]
        for j in range(i + 1, _M):
            mj = masks[j]
            if (mi & mj) == mi and mi != mj:
                cnt += 1
            elif (mi & mj) == mj:
                cnt += 1
    return cnt

def _family_from_masks(masks):
    return [[b for b in range(_N) if (m >> b) & 1] for m in masks]

def _base_family():
    # B_5: all 32 subsets of {0..4}
    base = []
    for m in range(1 << 5):
        base.append(m)
    # 8 extensions containing element 5, chosen by |T| descending
    # T = subset of {0..4}; set = {5} | T
    ext_T = [
        [0, 1, 2, 3, 4],  # size 5
        [0, 1, 2, 3],     # size 4
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
        [0, 1, 2],        # size 3
        [0, 1, 3],
    ]
    for T in ext_T:
        m = (1 << 5)
        for x in T:
            m |= (1 << x)
        base.append(m)
    return base  # 40 masks

def _local_search(masks, budget):
    """Hill-climb: replace one set with a new one that increases count."""
    best = _count_comparable(_family_from_masks(masks))
    cur = list(masks)
    cur_sets = set(cur)
    start = time.time()
    improved = True
    # candidate pool: all subsets of {0..11} is too large (4096); restrict to
    # subsets of {0..5} (64) plus random larger sets
    small_pool = list(range(1 << 6))
    import random
    rng = random.Random(12345)
    # build a diverse candidate pool
    cand_pool = list(set(small_pool))
    for _ in range(2000):
        r = rng.randint(0, 11)
        m = 0
        for x in range(12):
            if rng.random() < 0.35:
                m |= (1 << x)
        cand_pool.append(m)
    cand_pool = list(set(cand_pool))

    while improved and (time.time() - start) < budget:
        improved = False
        for i in range(_M):
            if (time.time() - start) > budget:
                break
            cur_sets.discard(cur[i])
            best_new = best
            best_new_m = cur[i]
            for cm in cand_pool:
                if cm in cur_sets:
                    continue
                # quick delta estimate: count pairs involving i
                # compute new count for position i vs all others
                mi = cm
                delta = 0
                for j in range(_M):
                    if j == i:
                        continue
                    mj = cur[j]
                    # old contribution
                    old = 1 if ((cur[i] & mj) == cur[i] and cur[i] != mj) or ((cur[i] & mj) == mj) else 0
                    new = 1 if ((mi & mj) == mi and mi != mj) or ((mi & mj) == mj) else 0
                    delta += new - old
                if best + delta > best_new:
                    best_new = best + delta
                    best_new_m = cm
            cur_sets.add(cur[i])
            if best_new > best:
                cur[i] = best_new_m
                cur_sets.discard(best_new_m)
                best = best_new
                improved = True
    return cur, best

def construct():
    t0 = time.time()
    masks = _base_family()
    base_score = _count_comparable(_family_from_masks(masks))

    # local search with time budget
    budget = max(1.0, _DEADLINE - (time.time() - t0) - 3.0)
    masks, score = _local_search(masks, budget)

    # ensure exactly 40 distinct, valid
    seen = set()
    out = []
    for m in masks:
        if m not in seen:
            seen.add(m)
            out.append(m)
    # if duplicates slipped, pad with fresh masks
    if len(out) < _M:
        for m in range(1 << _N):
            if len(out) >= _M:
                break
            if m not in seen:
                seen.add(m)
                out.append(m)
    out = out[:_M]

    result = _family_from_masks(out)
    # type sanity: ints and lists
    return [[int(x) for x in s] for s in result]
# EVOLVE-BLOCK-END
