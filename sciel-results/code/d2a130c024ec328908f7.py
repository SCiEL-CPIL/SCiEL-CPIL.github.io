import itertools
import time
import random
import numpy as np
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all t-subsets and their ids
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_id = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets and their ids
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    ksub_to_id = {ks: i for i, ks in enumerate(all_ksubs)}

    # coverage[ksub_id, tsub_id] = 1 if tsub is contained in ksub
    coverage = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for ks in all_ksubs:
        cid = ksub_to_id[ks]
        for ts in itertools.combinations(ks, t):
            coverage[cid, tsub_to_id[ts]] = 1

    def greedy_set_cover_fast(seed=0):
        rng = random.Random(seed)
        remaining = np.ones(n_tsubs, dtype=np.uint8)
        remaining_count = n_tsubs
        chosen = []
        used = np.zeros(n_ksubs, dtype=np.bool_)

        while remaining_count > 0 and time.time() < deadline - 25:
            scores = coverage @ remaining  # (n_ksubs,)
            scores[used] = -1
            best_id = int(np.argmax(scores))
            best_score = int(scores[best_id])
            if best_score <= 0:
                break
            used[best_id] = True
            chosen.append(all_ksubs[best_id])
            remaining = np.where(coverage[best_id] == 1, 0, remaining)
            remaining_count = int(np.count_nonzero(remaining))
        return chosen

    def local_search(blocks, deadline):
        # Build coverage count
        count = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            count += coverage[ksub_to_id[tuple(b)]]

        # Remove redundant blocks iteratively
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            new_blocks = []
            for b in blocks:
                cid = ksub_to_id[tuple(b)]
                only = (count == 1) & (coverage[cid] == 1)
                if not np.any(only):
                    count -= coverage[cid]
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks

        # 2-for-1 swap
        n = len(blocks)
        if n >= 2:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                # rebuild count
                count = np.zeros(n_tsubs, dtype=np.int32)
                for b in blocks:
                    count += coverage[ksub_to_id[tuple(b)]]
                n = len(blocks)
                for i in range(n):
                    if time.time() > deadline - 3:
                        break
                    ci = ksub_to_id[tuple(blocks[i])]
                    for j in range(i + 1, n):
                        if time.time() > deadline - 3:
                            break
                        cj = ksub_to_id[tuple(blocks[j])]
                        # tsubs that would become uncovered
                        orphaned = (count == 1) & ((coverage[ci] == 1) | (coverage[cj] == 1))
                        if not np.any(orphaned):
                            # safe: remove both
                            blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                            improved = True
                            break
                    if improved:
                        break
        return blocks

    best_blocks = None
    best_count = float('inf')

    for seed in range(20):
        if time.time() > deadline - 30:
            break
        blocks = greedy_set_cover_fast(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is not None:
        best_blocks = local_search(best_blocks, deadline - 3)
        best_count = len(best_blocks)

    if best_blocks is None:
        best_blocks = greedy_set_cover_fast(0)

    return best_blocks

if __name__ == "__main__":
    pass
