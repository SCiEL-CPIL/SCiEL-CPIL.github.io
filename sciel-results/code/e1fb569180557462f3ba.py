import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start = time.time()
    DEADLINE = 118.0  # safety margin under the 150s kill
    MAX_BLOCKS = int(v * (v - 1) * (v - 2) / (t * (t - 1) * (t - 2))) + 10  # C(v,t) cap

    def time_ok():
        return time.time() < DEADLINE - 2.0

    # Precompute all t-combinations and their index map
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    comb_to_idx = {c: i for i, c in enumerate(all_t)}

    # tsub_start[c] = index of the t-subset (c, c+1, ..., c+t-1)
    tsub_start = np.empty(v, dtype=np.int64)
    for c in range(0, v - t + 1):
        tsub_start[c] = comb_to_idx[(c, c + 1, c + t - 1)]

    # Candidate index arrays for fast numpy gain computation
    cand_idx = np.arange(v, dtype=np.int64)

    def greedy(seed_rng, min_gain):
        rng = random.Random(seed_rng)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any() and len(blocks) < MAX_BLOCKS:
            if not time_ok():
                return None
            # Pick an uncovered t-subset to start the block from
            unc_idx = np.where(uncovered)[0]
            start_idx = unc_idx[rng.randrange(len(unc_idx))]
            block = list(all_t[start_idx])
            block_set = set(block)

            # Build the block greedily
            while len(block) < k:
                if not time_ok():
                    return None
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break

                # Vectorized gain computation
                if t - 1 > 0:
                    m = len(block)
                    if m == t - 1:
                        # only one (t-1)-subset: the block itself
                        minS = block[0]
                        g = (cand_idx >= minS) & (cand_idx <= minS + t)
                        idxs = tsub_start[minS] + (cand_idx - minS)
                        gains = np.where(g, np.where(uncovered[idxs], 1, 0), 0)
                    else:
                        # accumulate over all (t-1)-subsets of the partial block
                        acc = np.zeros(v, dtype=np.int8)
                        for sub in itertools.combinations(block, t - 1):
                            minS = sub[0]
                            g = (cand_idx >= minS) & (cand_idx <= minS + t)
                            idxs = tsub_start[minS] + (cand_idx - minS)
                            acc += np.where(g, np.where(uncovered[idxs], 1, 0), 0)
                        gains = acc
                    maxg = int(gains.max())
                    if maxg == 0:
                        # No gain; pick a random candidate
                        block.append(candidates[rng.randrange(len(candidates))])
                    else:
                        # Random among candidates with max gain (>= min_gain)
                        thr = min_gain if min_gain > 0 else 0
                        pool = [x for x in candidates if gains[x] >= max(thr, maxg)]
                        block.append(pool[rng.randrange(len(pool))])
                else:
                    # t == 1 edge case
                    block.append(candidates[rng.randrange(len(candidates))])
                block_set.add(block[-1])

            block.sort()
            blocks.append(block)

            # Mark all t-subsets of this block as covered
            for tsub in itertools.combinations(block, t):
                uncovered[comb_to_idx[tsub]] = False

        if uncovered.any():
            return None  # incomplete
        return blocks

    best = None
    attempt = 0
    while time.time() < DEADLINE - 4.0:
        if best is not None and len(best) <= 76:
            break  # can't do better than known best of 76
        # Vary the greedy strength across attempts for diversity
        min_gain = 0 if attempt % 3 == 0 else 1
        res = greedy(attempt, min_gain)
        if res is not None:
            if best is None or len(res) < len(best):
                best = res
        attempt += 1

    if best is None:
        # Fallback: a single deterministic greedy (no random) to guarantee a valid answer
        rng = random.Random(0)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        while uncovered.any() and len(blocks) < MAX_BLOCKS:
            unc_idx = np.where(uncovered)[0]
            start_idx = unc_idx[0]
            block = list(all_t[start_idx])
            block_set = set(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                if t - 1 > 0:
                    m = len(block)
                    if m == t - 1:
                        minS = block[0]
                        g = (cand_idx >= minS) & (cand_idx <= minS + t)
                        idxs = tsub_start[minS] + (cand_idx - minS)
                        gains = np.where(g, np.where(uncovered[idxs], 1, 0), 0)
                    else:
                        acc = np.zeros(v, dtype=np.int8)
                        for sub in itertools.combinations(block, t - 1):
                            minS = sub[0]
                            g = (cand_idx >= minS) & (cand_idx <= minS + t)
                            idxs = tsub_start[minS] + (cand_idx - minS)
                            acc += np.where(g, np.where(uncovered[idxs], 1, 0), 0)
                        gains = acc
                    maxg = int(gains.max())
                    if maxg == 0:
                        block.append(candidates[0])
                    else:
                        best_ci = int(np.argmax(gains))
                        block.append(candidates[best_ci])
                else:
                    block.append(candidates[0])
                block_set.add(block[-1])
            block.sort()
            blocks.append(block)
            for tsub in itertools.combinations(block, t):
                uncovered[comb_to_idx[tsub]] = False
        best = blocks

    return best
