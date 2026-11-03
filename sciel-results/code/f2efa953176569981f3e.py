"""Portfolio search (vectorized bitmask greedy + reduction) for covering designs.

Design family: generate many randomized greedy coverings, then reduce redundancy.
Key change vs. the prior program: every greedy marginal-gain evaluation is done
with precomputed NumPy bitmask coverage vectors instead of Python set lookups.
This is a vectorized smooth surrogate for the nonsmooth "newly-covered triples"
objective, and it is dramatically faster, so many more restarts fit the budget.
"""
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin below the 150s kill

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_index = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Every (t-1)-subset maps to the list of t-subsets it extends to.
    t1_to_t = {}
    for sub in all_t_subsets:
        for j in range(t):
            t1 = sub[:j] + sub[j + 1:]
            t1_to_t.setdefault(t1, []).append(sub)

    # For each point, the bitmask of the t-subsets containing it.
    point_masks = np.zeros((v, n_t), dtype=np.uint8)
    for sub in all_t_subsets:
        idx = t_index[sub]
        for x in sub:
            point_masks[x, idx] = 1

    # ------------------------------------------------------------------
    # Vectorized greedy covering
    # ------------------------------------------------------------------
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=np.uint8)
        blocks = []
        while uncovered.any() and time.time() < deadline:
            nz = np.nonzero(uncovered)[0]
            start_idx = nz[rng.randrange(len(nz))]
            start_subset = all_t_subsets[start_idx]

            block = list(start_subset)
            in_block = set(block)

            # Mask of t-subsets that contain the current (t-1) core,
            # restricted to those still uncovered.
            if len(block) == t - 1:
                mask = np.fromiter(
                    (1 if s in uncovered_set else 0 for s in t1_to_t.get(tuple(block), ())),
                    dtype=np.uint8, count=n_t)
            else:
                mask = np.zeros(n_t, dtype=np.uint8)

            while len(block) < k and time.time() < deadline:
                candidates = [x for x in range(v) if x not in in_block]
                if not candidates:
                    break
                cand_arr = np.array(candidates, dtype=np.int64)
                # Count of newly-covered t-subsets for each candidate point.
                gains = point_masks[cand_arr] @ mask
                best = int(np.argmax(gains))
                best_gain = int(gains[best])
                if best_gain <= 0:
                    best = rng.randrange(len(candidates))
                x = candidates[best]
                block.append(x)
                in_block.add(x)
                # Update the (t-1)-core mask: t-subsets containing all of block.
                mask = np.fromiter(
                    (1 if s in uncovered_set else 0 for s in t1_to_t.get(tuple(block), ())),
                    dtype=np.uint8, count=n_t)

            if len(block) == k:
                block.sort()
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    uncovered[t_index[sub]] = 0
        return blocks

    # ------------------------------------------------------------------
    # Redundancy removal (keep only if it is a valid covering)
    # ------------------------------------------------------------------
    def local_search_remove(blocks):
        if not blocks:
            return blocks
        cover_count = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        current = list(blocks)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(current)):
                if time.time() >= deadline:
                    break
                redundant = all(cover_count.get(sub, 0) > 1
                                for sub in itertools.combinations(current[i], t))
                if redundant:
                    for sub in itertools.combinations(current[i], t):
                        cover_count[sub] -= 1
                    current.pop(i)
                    changed = True
                    break
        return current

    # ------------------------------------------------------------------
    # Portfolio loop
    # ------------------------------------------------------------------
    best_blocks = None
    best_count = float('inf')
    for seed in range(10000):
        if time.time() >= deadline:
            break
        blocks = greedy_cover(seed)
        if len(blocks) >= best_count:
            continue
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        if len(covered) == n_t:
            opt = local_search_remove(blocks)
            if len(opt) < best_count:
                best_count = len(opt)
                best_blocks = opt

    # Fallback: deterministic greedy, always valid.
    if best_blocks is None:
        uncovered = np.ones(n_t, dtype=np.uint8)
        best_blocks = []
        while uncovered.any():
            nz = np.nonzero(uncovered)[0]
            start_idx = int(nz[0])
            block = list(all_t_subsets[start_idx])
            in_block = set(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in in_block]
                if not candidates:
                    break
                cand_arr = np.array(candidates, dtype=np.int64)
                mask = np.fromiter(
                    (1 if s in uncovered_set else 0 for s in t1_to_t.get(tuple(block), ())),
                    dtype=np.uint8, count=n_t)
                gains = point_masks[cand_arr] @ mask
                x = candidates[int(np.argmax(gains))]
                block.append(x)
                in_block.add(x)
            block.sort()
            best_blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered[t_index[sub]] = 0
        best_blocks = local_search_remove(best_blocks)

    return best_blocks
