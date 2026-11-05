"""Max-SAT / WalkSAT-style local search for the K(2,2,2)-free triple system on 16 points."""
import random
import time
import numpy as np

# ---------- Precomputation (module level, one-time cost) ----------
N = 16
ALL = 0
for _i in range(N):
    ALL |= 1 << _i

# All C(16,3) triples
TRIPLES = []
for _a in range(N):
    for _b in range(_a + 1, N):
        for _c in range(_b + 1, N):
            TRIPLES.append((_a, _b, _c))
T = len(TRIPLES)  # 560

# For each triple, the set of points it uses (16-bit)
TRIPLE_MASK = [0] * T
for _i, (_a, _b, _c) in enumerate(TRIPLES):
    TRIPLE_MASK[_i] = (1 << _a) | (1 << _b) | (1 << _c)

# Point -> list of triple indices containing it
POINT_TRIPLES = [[] for _ in range(N)]
for _i in range(T):
    for _p in TRIPLES[_i]:
        POINT_TRIPLES[_p].append(_i)

# Enumerate all K(2,2,2) configurations: three disjoint pairs.
# Each config = 8 transversal triples.
def _all_configs():
    configs = []
    seen = set()
    points = list(range(N))
    # choose 6 distinct points, partition into 3 ordered pairs
    import itertools
    for six in itertools.combinations(points, 6):
        # all ways to split 6 into 3 ordered pairs
        for perm in itertools.permutations(six):
            pairs = (perm[0:2], perm[2:4], perm[4:6])
            # canonical form to dedupe
            cp = tuple(sorted(tuple(sorted(p)) for p in pairs))
            if cp in seen:
                continue
            seen.add(cp)
            p1, p2, p3 = pairs
            # 8 transversal triples
            tset = set()
            for x in p1:
                for y in p2:
                    for z in p3:
                        tset.add((x, y, z))
            configs.append(frozenset(tset))
    return configs

_CONFIGS = _all_configs()
C = len(_CONFIGS)

# For each config, the 8 triple indices
CFG_TRIPLES = []
for _cs in _CONFIGS:
    idx = []
    for _t in _cs:
        # find index
        idx.append(TRIPLES.index(_t))
    CFG_TRIPLES.append(idx)

# For each triple, which configs does it complete (bit over configs)
TRIPLE_COMPLETES = [0] * T
for _ci, _idx in enumerate(CFG_TRIPLES):
    bit = 1 << _ci
    for _ti in _idx:
        TRIPLE_COMPLETES[_ti] |= bit

# numpy arrays for fast bad-count
_CFG_TRIPLES_ARR = np.array(CFG_TRIPLES, dtype=np.int64)          # (C, 8)
_CFG_TRIPLES_BYTES = np.packbits(_CFG_TRIPLES_ARR.reshape(-1)).reshape(C, 8)
_CFG_TRIPLES_BYTES_U8 = _CFG_TRIPLES_BYTES.astype(np.uint8)
_TRIPLE_COMPLETES_U64 = np.array(TRIPLE_COMPLETES, dtype=np.uint64)

# ---------- Search ----------
def _count_bad(selected):
    """Return number of configs fully covered by selected (bit over triples)."""
    sel_bytes = np.packbits(selected.reshape(-1)).astype(np.uint8)   # (T,8)
    # For each config, AND its 8 rows -> (C,8); check all bytes == 255
    covered = np.all(
        np.all(sel_bytes[_CFG_TRIPLES_ARR] == 255, axis=2), axis=1
    )
    return int(np.count_nonzero(covered))


def _greedy_construct(rng):
    """Build a valid set via randomized greedy add."""
    selected = np.zeros(T, dtype=bool)
    order = list(range(T))
    while True:
        rng.shuffle(order)
        # try to add as many as possible
        added = 0
        for ti in order:
            if selected[ti]:
                continue
            # would adding ti create a bad config?
            # a config becomes bad iff all its other 7 members already selected
            # and ti is one of them. Check: for each config containing ti,
            # are all other 7 selected?
            ok = True
            for ci, idx in enumerate(CFG_TRIPLES):
                if ti not in idx:
                    continue
                # check all other 7
                others = [j for j in idx if j != ti]
                if all(selected[j] for j in others):
                    ok = False
                    break
            if ok:
                selected[ti] = True
                added += 1
        if added == 0:
            break
        # if we made progress, we could continue; but greedy single pass is fine
        break
    return selected


def construct():
    start = time.time()
    DEADLINE = start + 120.0   # safety margin before 150s kill
    rng = random.Random(12345)

    best_selected = None
    best_count = -1

    # Initial: greedy construction
    selected = _greedy_construct(rng)
    cnt = int(np.count_nonzero(selected))
    bad = _count_bad(selected)
    if bad == 0 and cnt > best_count:
        best_count = cnt
        best_selected = selected.copy()

    # WalkSAT-style local search
    # State: selected (bool array), bad set (set of config indices)
    def _bad_set(selected):
        sel_bytes = np.packbits(selected.reshape(-1)).astype(np.uint8)
        covered = np.all(
            np.all(sel_bytes[_CFG_TRIPLES_ARR] == 255, axis=2), axis=1
        )
        return set(np.nonzero(covered)[0].tolist())

    cur_bad = _bad_set(selected)

    no_improve = 0
    max_no_improve = 5000
    while time.time() < DEADLINE:
        if not cur_bad:
            # valid; try to add a triple
            candidates = [ti for ti in range(T) if not selected[ti]]
            if not candidates:
                break
            # pick a candidate that doesn't complete any config
            good = []
            for ti in candidates:
                # check if adding ti completes any config
                completes = False
                for ci in cur_bad:  # empty, so skip
                    pass
                # need to check configs containing ti where other 7 are selected
                for ci, idx in enumerate(CFG_TRIPLES):
                    if ti not in idx:
                        continue
                    others = [j for j in idx if j != ti]
                    if all(selected[j] for j in others):
                        completes = True
                        break
                if not completes:
                    good.append(ti)
            if good:
                ti = rng.choice(good)
                selected[ti] = True
                # update bad set incrementally: only configs containing ti
                for ci, idx in enumerate(CFG_TRIPLES):
                    if ti not in idx:
                        continue
                    if all(selected[j] for j in idx):
                        cur_bad.add(ci)
                    elif ci in cur_bad:
                        cur_bad.discard(ci)
                cnt = int(np.count_nonzero(selected))
                if not cur_bad and cnt > best_count:
                    best_count = cnt
                    best_selected = selected.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # stuck; break a bad config by removing a triple
                if cur_bad:
                    ci = rng.choice(list(cur_bad))
                    idx = CFG_TRIPLES[ci]
                    # remove one of the 8 (prefer one that appears in few configs)
                    idx_list = sorted(idx, key=lambda j: bin(_TRIPLE_COMPLETES_U64[j]).count('1'))
                    ti = idx_list[0]
                    selected[ti] = False
                    for cj, cidx in enumerate(CFG_TRIPLES):
                        if ti not in cidx:
                            continue
                        if all(selected[j] for j in cidx):
                            cur_bad.add(cj)
                        elif cj in cur_bad:
                            cur_bad.discard(cj)
                    no_improve += 1
                else:
                    no_improve += 1

            if no_improve > max_no_improve:
                # restart with a fresh greedy
                selected = _greedy_construct(rng)
                cur_bad = _bad_set(selected)
                cnt = int(np.count_nonzero(selected))
                if not cur_bad and cnt > best_count:
                    best_count = cnt
                    best_selected = selected.copy()
                no_improve = 0
                max_no_improve = max(5000, max_no_improve // 2)
        else:
            # has bad configs; remove a triple from a random bad config
            ci = rng.choice(list(cur_bad))
            idx = CFG_TRIPLES[ci]
            idx_list = sorted(idx, key=lambda j: bin(_TRIPLE_COMPLETES_U64[j]).count('1'))
            ti = idx_list[rng.randint(0, len(idx_list) - 1)]
            selected[ti] = False
            for cj, cidx in enumerate(CFG_TRIPLES):
                if ti not in cidx:
                    continue
                if all(selected[j] for j in cidx):
                    cur_bad.add(cj)
                elif cj in cur_bad:
                    cur_bad.discard(cj)
            no_improve += 1
            if no_improve > max_no_improve:
                selected = _greedy_construct(rng)
                cur_bad = _bad_set(selected)
                cnt = int(np.count_nonzero(selected))
                if not cur_bad and cnt > best_count:
                    best_count = cnt
                    best_selected = selected.copy()
                no_improve = 0
                max_no_improve = max(5000, max_no_improve // 2)

    # Final: ensure we return a valid (no bad) set; if best is None, fall back
    if best_selected is None:
        # return a safe small valid set
        sel = np.zeros(T, dtype=bool)
        for ti in range(T):
            ok = True
            for ci, idx in enumerate(CFG_TRIPLES):
                if ti not in idx:
                    continue
                others = [j for j in idx if j != ti]
                if all(sel[j] for j in others):
                    ok = False
                    break
            if ok:
                sel[ti] = True
        best_selected = sel

    # Convert to list of triples
    result = []
    for ti in range(T):
        if best_selected[ti]:
            result.append(list(TRIPLES[ti]))
    return result
