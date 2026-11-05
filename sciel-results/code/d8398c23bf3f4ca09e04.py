# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _cover_counts(blocks, v, t):
    """Count how many blocks cover each t-subset. Returns dict {subset_tuple: int}."""
    counts = {}
    for b in blocks:
        for sub in itertools.combinations(b, t):
            counts[sub] = counts.get(sub, 0) + 1
    return counts


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before the 150s kill

    # Precompute all candidate (k-1)-subsets of the v points.
    cand = np.array(list(itertools.combinations(range(v), k - 1)), dtype=np.int64)
    n_cand = len(cand)
    # Precompute, for each candidate, the t-subsets it covers (the (k-1)-set plus x).
    cand_sets = []
    for c in cand:
        cs = set()
        for x in range(v):
            if x not in c:
                cs.add(tuple(sorted(c + (x,))))
        cand_sets.append(cs)

    def _greedy(allowed, seed):
        """One greedy pass. Returns list of blocks (valid, may be incomplete if deadline hit)."""
        rng = random.Random(seed)
        counts = {}
        blocks = []
        covered = 0
        total = 0
        for s in itertools.combinations(range(v), t):
            if s in allowed:
                total += 1
        while covered < total:
            if time.time() >= deadline:
                return blocks
            # Most constrained uncovered t-subset: fewest candidate supersets.
            best = None
            for s in itertools.combinations(range(v), t):
                if s in allowed and counts.get(s, 0) == 0:
                    cnt = sum(1 for cs in cand_sets if s in cs)
                    if best is None or cnt < best[0]:
                        best = (cnt, s)
                        if cnt == 1:
                            break
            s = best[1]
            # Candidate blocks: (k-1)-sets that cover s, with all points in allowed.
            valid_idx = [i for i, cs in enumerate(cand_sets)
                         if s in cs and all(p in allowed for p in cand[i])]
            if not valid_idx:
                # Fallback: any candidate with allowed points (should not happen if s is uncovered)
                valid_idx = [i for i in range(n_cand) if all(p in allowed for p in cand[i])]
            if not valid_idx:
                return blocks
            # Gain per candidate: number of uncovered t-subsets it would cover.
            gains = np.zeros(len(valid_idx), dtype=np.int64)
            for pos, idx in enumerate(valid_idx):
                for sub in cand_sets[idx]:
                    if counts.get(sub, 0) == 0:
                        gains[pos] += 1
            j = int(np.argmax(gains))
            chosen = cand[valid_idx[j]]
            # Choose the extra point x that maximizes additional uncovered coverage.
            best_x = None
            best_gain = -1
            for x in range(v):
                if x not in allowed or x in chosen:
                    continue
                g = 0
                for sub in cand_sets[valid_idx[j]]:
                    if x in sub and counts.get(sub, 0) == 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                return blocks
            block = tuple(sorted(chosen + (best_x,)))
            blocks.append(list(block))
            for sub in itertools.combinations(block, t):
                if counts.get(sub, 0) == 0:
                    covered += 1
                counts[sub] = counts.get(sub, 0) + 1
        return blocks

    # Single pass (allowed = all points) is always valid when complete.
    base = _greedy(set(range(v)), 0)
    if len(base) == 0:
        # Fallback: trivial covering (one block per t-subset) if nothing worked.
        return [list(s) + list(range(v, v + 0)) for s in []] or \
               [[i for i in range(v)] for _ in range(0)] or \
               [list(s) for s in itertools.combinations(range(v), t)]

    # Multi-start with random allowed-sets to escape local optima.
    best = base
    best_len = len(base)
    seed = 1
    while time.time() < deadline - 5.0 and len(best) > 0:
        # Randomly drop a small fraction of points to force diversity.
        drop = random.Random(seed).random() * 0.25
        allowed = set(range(v))
        if drop > 0:
            for p in range(v):
                if random.Random(seed + p).random() < drop:
                    allowed.discard(p)
        if len(allowed) < k:
            allowed = set(range(v))
        cand_blocks = _greedy(allowed, seed)
        # Only accept if it's a complete valid cover and shorter.
        if len(cand_blocks) < best_len:
            cnt = _cover_counts(cand_blocks, v, t)
            if all(cnt.get(s, 0) >= 1 for s in itertools.combinations(range(v), t)):
                best = cand_blocks
                best_len = len(cand_blocks)
        seed += 1
    return best
# EVOLVE-BLOCK-END
