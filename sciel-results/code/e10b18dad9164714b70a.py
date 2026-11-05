import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0

    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    n_all_blocks = len(all_k_blocks)

    block_triplets = []
    for b in all_k_blocks:
        tids = [triplet_to_id[triplet] for triplet in itertools.combinations(b, t)]
        block_triplets.append(tids)

    triple_blocks = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplets):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    def greedy_ratio(v, k, t, seed=0, alpha=1.0, beta=0.0):
        rng = random.Random(seed)
        uncovered = np.ones(n_triplets, dtype=np.int32)
        gain = np.array([len(ts) for ts in block_triplets], dtype=np.int32)
        # For ratio: gain / (number of points in block) = gain/k, constant k
        # But we can add a "redundancy penalty": penalize blocks that overlap heavily with already-selected ones
        # Alternative: use gain / (1 + overlap_score) where overlap_score measures how much this block overlaps with selected blocks
        # Simpler: use gain - beta * (number of triplets already covered by other selected blocks in this block)
        # Actually, the standard greedy already does this via gain.
        # Let's try: score = gain - beta * (k - gain) to penalize blocks with many redundant points
        # Or: score = gain / (k - gain + 1) to penalize redundancy

        selected_blocks = []
        selected_set = set()

        while uncovered.sum() > 0:
            if time.time() > deadline - 15:
                while uncovered.sum() > 0:
                    b_idx = rng.randint(0, n_all_blocks - 1)
                    selected_blocks.append(b_idx)
                    for t_idx in block_triplets[b_idx]:
                        if uncovered[t_idx] > 0:
                            uncovered[t_idx] = 0
                            for other_b_idx in triple_blocks[t_idx]:
                                gain[other_b_idx] -= 1
                break

            if alpha > 0 and beta > 0:
                # score = alpha * gain - beta * (k - gain) = (alpha + beta) * gain - beta * k
                # This is equivalent to maximizing gain with a constant offset, so just max gain
                # Better: score = gain / (k - gain + 1)
                # Or use a soft penalty
                scores = gain.astype(np.float64)
                # Penalize blocks where many points are already "wasted"
                # For each block, count how many of its triplets are already covered
                # Actually, gain already accounts for this. Let's try a different approach:
                # Use gain / (number of unique points in block not in selected_set)
                # But that's expensive. Let's just use max gain with random tie-breaking
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = rng.choice(candidates.tolist())
            else:
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = rng.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            for t_idx in block_triplets[b_idx]:
                if uncovered[t_idx] > 0:
                    uncovered[t_idx] -= 1
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return [all_k_blocks[b_idx] for b_idx in selected_blocks]

    def greedy_pointwise(v, k, t, seed=0, bias=0.0):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []

        while uncovered:
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]

            block = list(start_triplet)
            block_set = set(block)

            while len(block) < k:
                pairs_in_block = list(itertools.combinations(block, t - 1))

                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1

                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = triplet_to_id[triplet]
                        if tid in uncovered:
                            gain += 1
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                uncovered.discard(tid)

        return blocks

    # Phase 1: Multiple greedy constructions
    best_blocks = None
    best_count = float('inf')

    for seed in range(30):
        if time.time() > deadline - 50:
            break
        blocks = greedy_ratio(v, k, t, seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    for seed in range(15):
        if time.time() > deadline - 30:
            break
        blocks = greedy_pointwise(v, k, t, seed, bias=0.2)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_ratio(v, k, t, 0)

    # Phase 2: Local search
    count = np.zeros(n_triplets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[triplet_to_id[triplet]] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        block_triplet_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(triplet_to_id[triplet])
            block_triplet_ids.append(ids)

        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            for j in range(i + 1, n):
                ids_b1 = block_triplet_ids[i]
                ids_b2 = block_triplet_ids[j]

                critical_points = set()

                for tid in ids_b1:
                    c = count[tid]
                    covered_by_b1 = 1
                    covered_by_b2 = 1 if tid in ids_b2 else 0
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])

                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    covered_by_b1 = 0
                    covered_by_b2 = 1
                    if c == covered_by_b1 + covered_by_b2:
                        critical_points.update(all_triplets[tid])

                if not critical_points:
                    update_counts([blocks[i], blocks[j]], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                if len(critical_points) <= k:
                    new_block = list(critical_points)
                    while len(new_block) < k:
                        p = random.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    update_counts([blocks[i], blocks[j]], -1)
                    update_counts([new_block], 1)

                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    progress_made = True
    while progress_made and time.time() < deadline - 3:
        progress_made = False

        current_blocks, removed = try_remove_redundant(current_blocks)
        if removed:
            progress_made = True

        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True

    current_blocks, _ = try_remove_redundant(current_blocks)

    return current_blocks

if __name__ == "__main__":
    pass
