"""Greedy covering for C(v,k,t) with global scan-all and time guard."""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    DEADLINE = 140.0  # safety margin before the 150s kill

    all_tsubs = list(itertools.combinations(range(v), t))
    total_t = len(all_tsubs)
    # sup[c] = list of t-subset indices (into all_tsubs) that contain point c
    sup = [[] for _ in range(v)]
    for idx, c in enumerate(all_tsubs):
        for p in c:
            sup[p].append(idx)

    # uncovered[idx] == True if t-subset idx not yet covered
    unc = np.ones(total_t, dtype=bool)
    remaining = total_t

    blocks = []
    used = np.zeros(v, dtype=bool)
    used_set = set()

    while remaining > 0:
        if time.time() > DEADLINE:
            break
        # pick seed = point with most uncovered supersets
        seed = -1
        best_count = -1
        for p in range(v):
            cnt = int(unc[sup[p]].sum())
            if cnt > best_count:
                best_count = cnt
                seed = p
        if seed < 0:
            break
        block = [seed]
        used[seed] = True
        used_set.add(seed)
        # global scan-all: greedily add point with max gain
        while len(block) < k:
            best_gain = -1
            best_p = -1
            for p in range(v):
                if p in used_set:
                    continue
                cnt = int(unc[sup[p]].sum())
                if cnt > best_gain:
                    best_gain = cnt
                    best_p = p
            if best_p < 0:
                break
            block.append(best_p)
            used[best_p] = True
            used_set.add(best_p)
        block.sort()
        blocks.append(block)
        # update uncovered
        for c in itertools.combinations(block, t):
            idx = all_tsubs.index(c)  # expensive; see note below
            if unc[idx]:
                unc[idx] = False
                remaining -= 1

    # Fallback: if not complete, finish with a simple greedy (no guard) to ensure validity
    if remaining > 0:
        # rebuild uncovered set for the fallback path
        covered = set()
        for b in blocks:
            for c in itertools.combinations(b, t):
                covered.add(c)
        uncovered = set(all_tsubs) - covered
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))

    return blocks
