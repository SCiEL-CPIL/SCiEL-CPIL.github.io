import numpy as np
import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135

    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)

    block_tsub_indices = np.empty((num_kblocks, 56), dtype=np.int32)
    for i, block in enumerate(all_kblocks):
        for j, tsub in enumerate(itertools.combinations(block, t)):
            block_tsub_indices[i, j] = tsub_to_idx[tsub]

    def greedy_run(rng, time_budget):
        uncovered = np.ones(num_tsubs, dtype=np.int8)
        chosen = []
        used = set()
        while True:
            if time.time() > time_budget:
                break
            if not uncovered.any():
                break
            gains = uncovered[block_tsub_indices].sum(axis=1)
            for bi in used:
                gains[bi] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            if rng.random() < 0.4:
                noise = rng.random(num_kblocks) * 0.0005
                perturbed = gains.copy()
                perturbed[gains >= 0] += noise[gains >= 0]
                perturbed[gains < 0] = -1
                best = int(np.argmax(perturbed))
            else:
                best = int(np.argmax(gains))
            chosen.append(best)
            used.add(best)
            uncovered[block_tsub_indices[best]] = 0
        return chosen

    def local_search(chosen, time_budget):
        chosen = list(chosen)
        cov_count = np.zeros(num_tsubs, dtype=np.int32)
        for bi in chosen:
            cov_count[block_tsub_indices[bi]] += 1

        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # Remove redundant
            new_chosen = []
            for bi in chosen:
                sub = block_tsub_indices[bi]
                if (cov_count[sub] > 1).all():
                    cov_count[sub] -= 1
                    improved = True
                else:
                    new_chosen.append(bi)
            chosen = new_chosen

            # Try swaps: for each chosen block, find unused block covering its "critical" tsubs
            if time.time() > time_budget - 5:
                break
            chosen_set = set(chosen)
            new_chosen2 = []
            for bi in chosen:
                sub = block_tsub_indices[bi]
                critical = sub[cov_count[sub] == 1]
                if len(critical) == 0:
                    new_chosen2.append(bi)
                    continue
                # Find unused block covering all critical
                # This is expensive; do a limited search
                found = False
                # Use a fast check: for each unused block, see if it covers all critical
                # Precompute: for each t-subset, which blocks cover it? Too big.
                # Instead, just skip swaps for now.
                new_chosen2.append(bi)
            chosen = new_chosen2

        return chosen

    def is_valid(chosen):
        cov = np.zeros(num_tsubs, dtype=np.int8)
        for bi in chosen:
            cov[block_tsub_indices[bi]] = 1
        return cov.all()

    rng = random.Random(42)
    best_chosen = None
    best_len = float('inf')

    # First greedy
    chosen = greedy_run(rng, deadline - 80)
    if is_valid(chosen):
        chosen = local_search(chosen, deadline - 50)
        if is_valid(chosen) and len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = list(chosen)

    # Randomized restarts
    for r in range(30):
        if time.time() > deadline - 30:
            break
        gb = deadline - 30 - (30 - r) * 1.5
        if gb < start_time + 3:
            break
        chosen = greedy_run(rng, gb)
        if not is_valid(chosen):
            continue
        chosen = local_search(chosen, deadline - 15)
        if is_valid(chosen) and len(chosen) < best_len:
            best_len = len(chosen)
            best_chosen = list(chosen)

    if best_chosen is not None and time.time() < deadline - 5:
        best_chosen = local_search(best_chosen, deadline - 2)
        if not is_valid(best_chosen):
            pass

    if best_chosen is None:
        return []

    return [list(all_kblocks[bi]) for bi in best_chosen]
