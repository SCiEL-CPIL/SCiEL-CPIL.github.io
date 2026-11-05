import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # leave ~35s safety margin

    # ---- Precomputation ----
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = all_triplets

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))

    # block_indices[bi, j] = triplet id covered by block bi
    block_indices = np.zeros((n_blocks, t_subsets_per_block), dtype=np.int32)
    for bi, block in enumerate(all_blocks):
        tids = [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        block_indices[bi] = tids

    # Frequency of each triplet across all blocks
    freq = np.bincount(block_indices.flatten(), minlength=n_triplets).astype(np.float64)
    inv_freq = 1.0 / (freq + 1e-9)

    # Inverse index: triplet_id -> list of block indices containing it
    inv_index = {}
    for bi in range(n_blocks):
        for tid in block_indices[bi]:
            tid_int = int(tid)
            if tid_int not in inv_index:
                inv_index[tid_int] = []
            inv_index[tid_int].append(bi)

    def get_triplet_ids(block):
        return [triplet_to_id[trip] for trip in itertools.combinations(block, t)]

    def blocks_containing(points):
        points_set = set(points)
        if len(points_set) > k:
            return
        remaining = [p for p in range(v) if p not in points_set]
        need = k - len(points_set)
        if need < 0:
            return
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points_set.union(comb)))

    # ---- Vectorized greedy set cover ----
    def vectorized_greedy(seed, mode='greedy'):
        rng = random.Random(seed)
        uncovered = np.ones(n_triplets, dtype=bool)
        selected = []

        while True:
            if time.time() > deadline - 30:
                break
            if mode == 'greedy':
                gains = uncovered[block_indices].sum(axis=1)
                best = int(np.argmax(gains))
            elif mode == 'weighted':
                w = np.where(uncovered, inv_freq, 0.0)
                gains = w[block_indices].sum(axis=1)
                best = int(np.argmax(gains))
            else:  # randomized
                gains = uncovered[block_indices].sum(axis=1)
                maxg = int(gains.max())
                if maxg == 0:
                    break
                cands = np.where(gains == maxg)[0]
                if len(cands) > 20:
                    cands = rng.sample(list(cands), 20)
                best = int(rng.choice(list(cands)))

            if gains[best] == 0:
                break
            selected.append(best)
            uncovered[block_indices[best]] = False
            if not uncovered.any():
                break

        return [list(all_blocks[i]) for i in selected]

    # ---- Soft surrogate penalty for local search ----
    def soft_penalty(counts, alpha):
        """
        Smooth surrogate: for each triplet, penalty = exp(-alpha * counts).
        Total penalty = sum of penalties. Lower is better.
        A block is 'soft-redundant' if removing it doesn't increase total penalty
        (i.e., all its triplets have counts >= 2, or the increase is small).
        """
        return np.exp(-alpha * counts).sum()

    def optimize(blocks, alpha_start=5.0, alpha_end=0.5, n_anneal_steps=5):
        """
        Local search with soft surrogate annealing.
        """
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for tid in get_triplet_ids(b):
                counts[tid] += 1

        def is_redundant_hard(block):
            for trip in itertools.combinations(block, t):
                if counts[triplet_to_id[trip]] == 1:
                    return False
            return True

        def is_soft_redundant(block, alpha):
            """
            Check if removing this block increases soft penalty by less than a threshold.
            """
            current_penalty = soft_penalty(counts, alpha)
            # Temporarily remove
            for tid in get_triplet_ids(block):
                counts[tid] -= 1
            new_penalty = soft_penalty(counts, alpha)
            # Restore
            for tid in get_triplet_ids(block):
                counts[tid] += 1
            # If penalty increase is very small, consider it redundant
            return (new_penalty - current_penalty) < 0.01

        def build_idsets(blocks):
            return [set(get_triplet_ids(b)) for b in blocks]

        max_iters = 8000
        n_anneal = n_anneal_steps
        for anneal_step in range(n_anneal):
            alpha = alpha_start - (alpha_start - alpha_end) * anneal_step / max(1, n_anneal - 1)
            iters_this_phase = max_iters // n_anneal
            for it in range(iters_this_phase):
                if time.time() > deadline - 8:
                    break
                improved = False
                n = len(blocks)
                idsets = build_idsets(blocks)

                # 1. Remove redundant blocks (hard first, then soft)
                for i in range(n):
                    if is_redundant_hard(blocks[i]):
                        b = blocks.pop(i)
                        for tid in get_triplet_ids(b):
                            counts[tid] -= 1
                        improved = True
                        break
                if not improved and alpha < 2.0:
                    for i in range(n):
                        if is_soft_redundant(blocks[i], alpha):
                            b = blocks.pop(i)
                            for tid in get_triplet_ids(b):
                                counts[tid] -= 1
                            improved = True
                            break
                if improved:
                    continue

                # 2. 2-for-1 replacement
                if n >= 2:
                    n_pairs = min(150, n * (n - 1) // 2)
                    tried = set()
                    for _ in range(n_pairs):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, n - 1)
                        if i == j or (i, j) in tried or (j, i) in tried:
                            continue
                        tried.add((i, j)); tried.add((j, i))
                        b1, b2 = blocks[i], blocks[j]
                        ids1, ids2 = idsets[i], idsets[j]
                        needed = [tid for tid in (ids1 | ids2) if counts[tid] == 1]
                        if not needed:
                            for idx in [j, i]:
                                br = blocks.pop(idx)
                                for tid in get_triplet_ids(br):
                                    counts[tid] -= 1
                            improved = True
                            break
                        pts = set()
                        for tid in needed:
                            pts.update(id_to_triplet[tid])
                        if len(pts) > k:
                            continue
                        for nb in blocks_containing(pts):
                            if set(nb) == set(b1) or set(nb) == set(b2):
                                continue
                            nid = set(get_triplet_ids(nb))
                            if all(tid in nid for tid in needed):
                                for idx in [j, i]:
                                    br = blocks.pop(idx)
                                    for tid in get_triplet_ids(br):
                                        counts[tid] -= 1
                                blocks.append(nb)
                                for tid in get_triplet_ids(nb):
                                    counts[tid] += 1
                                improved = True
                                break
                        if improved:
                            break

                # 3. 1-for-1 replacement
                if not improved and n > 0:
                    i = random.randint(0, n - 1)
                    b1 = blocks[i]
                    ids1 = idsets[i]
                    needed = [tid for tid in ids1 if counts[tid] == 1]
                    if needed:
                        pts = set()
                        for tid in needed:
                            pts.update(id_to_triplet[tid])
                        if len(pts) <= k:
                            for nb in blocks_containing(pts):
                                if set(nb) == set(b1):
                                    continue
                                nid = set(get_triplet_ids(nb))
                                if all(tid in nid for tid in needed):
                                    for tid in get_triplet_ids(b1):
                                        counts[tid] -= 1
                                    blocks[i] = nb
                                    for tid in get_triplet_ids(nb):
                                        counts[tid] += 1
                                    improved = True
                                    break

                # 4. 3-for-2 replacement
                if not improved and n >= 3:
                    n_tr = min(50, n * (n - 1) * (n - 2) // 6)
                    tried = set()
                    for _ in range(n_tr):
                        idxs = sorted(random.sample(range(n), 3))
                        key = tuple(idxs)
                        if key in tried:
                            continue
                        tried.add(key)
                        b1, b2, b3 = blocks[idxs[0]], blocks[idxs[1]], blocks[idxs[2]]
                        u = idsets[idxs[0]] | idsets[idxs[1]] | idsets[idxs[2]]
                        needed = [tid for tid in u if counts[tid] == 1]
                        if not needed:
                            for idx in [idxs[2], idxs[1], idxs[0]]:
                                br = blocks.pop(idx)
                                for tid in get_triplet_ids(br):
                                    counts[tid] -= 1
                            improved = True
                            break
                        pts = set()
                        for tid in needed:
                            pts.update(id_to_triplet[tid])
                        if len(pts) > k:
                            continue
                        for nb1 in blocks_containing(pts):
                            if set(nb1) in (set(b1), set(b2), set(b3)):
                                continue
                            nid1 = set(get_triplet_ids(nb1))
                            if not all(tid in nid1 for tid in needed):
                                continue
                            rem = [tid for tid in needed if tid not in nid1]
                            if not rem:
                                for idx in [idxs[2], idxs[1], idxs[0]]:
                                    br = blocks.pop(idx)
                                    for tid in get_triplet_ids(br):
                                        counts[tid] -= 1
                                blocks.append(nb1)
                                for tid in get_triplet_ids(nb1):
                                    counts[tid] += 1
                                improved = True
                                break
                            for nb2 in blocks_containing(pts):
                                if set(nb2) in (set(b1), set(b2), set(b3), set(nb1)):
                                    continue
                                nid2 = set(get_triplet_ids(nb2))
                                if all(tid in nid2 for tid in rem):
                                    for idx in [idxs[2], idxs[1], idxs[0]]:
                                        br = blocks.pop(idx)
                                        for tid in get_triplet_ids(br):
                                            counts[tid] -= 1
                                    blocks.append(nb1); blocks.append(nb2)
                                    for tid in get_triplet_ids(nb1):
                                        counts[tid] += 1
                                    for tid in get_triplet_ids(nb2):
                                        counts[tid] += 1
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

                if not improved:
                    break

        return blocks

    def ensure_valid(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for tid in get_triplet_ids(b):
                counts[tid] += 1
        uncovered = [i for i in range(n_triplets) if counts[i] == 0]
        while uncovered:
            if time.time() > deadline - 4:
                break
            tid = uncovered[0]
            best_block = None
            best_gain = -1
            for bi in inv_index.get(tid, []):
                gain = sum(1 for x in block_indices[bi] if counts[x] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block = all_blocks[bi]
            if best_block is None:
                break
            if best_block not in blocks:
                blocks.append(best_block)
            for tid2 in get_triplet_ids(best_block):
                counts[tid2] += 1
            uncovered = [i for i in range(n_triplets) if counts[i] == 0]
        return blocks

    def is_valid(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for tid in get_triplet_ids(b):
                counts[tid] += 1
        return int(np.min(counts)) > 0

    best_blocks = None
    best_len = float('inf')

    modes = ['greedy', 'weighted', 'randomized']
    seed = 0
    while time.time() < deadline - 35:
        mode = modes[seed % 3]
        blocks = vectorized_greedy(seed, mode)
        if not blocks:
            seed += 1
            continue
        opt = optimize(blocks)
        opt = ensure_valid(opt)
        if is_valid(opt) and len(opt) < best_len:
            best_len = len(opt)
            best_blocks = opt
        if best_len <= 76:
            break
        seed += 1

    if best_blocks is None:
        blocks = vectorized_greedy(0, 'greedy')
        best_blocks = ensure_valid(blocks)

    return best_blocks

if __name__ == "__main__":
    pass
