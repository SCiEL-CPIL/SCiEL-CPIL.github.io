"""Covering design C(v,k,t) via random-start greedy with local gain evaluation."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    N_t = len(all_t_subs)
    t_id = {s: i for i, s in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_k_blocks)

    block_t_ids = []
    for b in all_k_blocks:
        ids = [t_id[sub] for sub in itertools.combinations(b, t)]
        block_t_ids.append(ids)

    # Reverse incidence
    t_sub_to_blocks = [[] for _ in range(N_t)]
    for bi in range(n_blocks):
        for s in block_t_ids[bi]:
            t_sub_to_blocks[s].append(bi)

    def greedy_one_pass(seed):
        rng = random.Random(seed)
        unc = np.ones(N_t, dtype=bool)
        chosen = []
        
        while unc.any():
            unc_indices = np.where(unc)[0]
            # Pick random uncovered t-subset
            j = unc_indices[rng.randint(0, len(unc_indices) - 1)]
            # Candidate blocks: those containing t-subset j
            candidates = t_sub_to_blocks[j]
            # Among candidates, find one with max uncovered coverage
            best_idx = candidates[0]
            best_gain = -1
            for bi in candidates:
                gain = np.sum(unc[block_t_ids[bi]])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = bi
            chosen.append(best_idx)
            unc[block_t_ids[best_idx]] = False
        
        return chosen

    def reduce_blocks(chosen_indices, seed):
        rng = random.Random(seed)
        if not chosen_indices:
            return []
        cover_count = np.zeros(N_t, dtype=np.int32)
        for idx in chosen_indices:
            for s in block_t_ids[idx]:
                cover_count[s] += 1
        remaining = list(chosen_indices)
        changed = True
        while changed and time.time() < deadline - 8.0:
            changed = False
            order = list(range(len(remaining)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(remaining):
                    continue
                idx = remaining[pos]
                ids = block_t_ids[idx]
                for s in ids:
                    cover_count[s] -= 1
                redundant = True
                for s in ids:
                    if cover_count[s] == 0:
                        redundant = False
                        break
                if redundant:
                    remaining.pop(pos)
                    changed = True
                    break
                else:
                    for s in ids:
                        cover_count[s] += 1
        return remaining

    best_blocks = None
    best_count = float('inf')
    restart = 0
    while time.time() < deadline - 3.0:
        restart += 1
        seed = restart * 7919 + 3
        chosen = greedy_one_pass(seed)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen
        if time.time() < deadline - 12.0:
            reduced = reduce_blocks(best_blocks, seed=seed + 101)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced

    if best_blocks is None:
        best_blocks = greedy_one_pass(1)
    return [list(all_k_blocks[i]) for i in best_blocks]
