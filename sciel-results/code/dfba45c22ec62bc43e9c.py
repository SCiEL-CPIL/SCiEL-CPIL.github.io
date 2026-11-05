import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)

    block_triples = []
    for block in all_k_blocks:
        tids = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triples.append(tids)

    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, tids in enumerate(block_triples):
        for t_idx in tids:
            triple_blocks[t_idx].append(b_idx)

    triple_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_blocks]
    block_triples_np = [np.array(ts, dtype=np.int32) for ts in block_triples]

    def greedy_cover(seed, time_budget):
        rng = random.Random(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        selected = []

        while is_uncovered.any():
            if time.time() > time_budget:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())
            selected.append(b_idx)
            for t_idx in block_triples_np[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    gain[triple_blocks_np[t_idx]] -= 1
        return selected

    def complete_cover(selected, deadline):
        """If the greedy broke early, finish covering with a simple pass."""
        # Compute which triples are still uncovered
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1
        uncovered_mask = cover_count == 0

        while uncovered_mask.any():
            if time.time() > deadline:
                # Last resort: keep adding blocks until covered (ignore time)
                # But we must finish before hard deadline
                pass
            # Find block that covers most uncovered triples
            best_b = -1
            best_gain = 0
            # Sample some blocks to speed up
            sample_size = min(num_blocks, 2000)
            sample = random.sample(range(num_blocks), sample_size)
            for b_idx in sample:
                cnt = np.sum(uncovered_mask[block_triples_np[b_idx]])
                if cnt > best_gain:
                    best_gain = cnt
                    best_b = b_idx
            if best_b == -1:
                break
            selected.append(best_b)
            cover_count[block_triples_np[best_b]] += 1
            uncovered_mask[cover_count == 0] = False
            # Recompute uncovered
            uncovered_mask = cover_count == 0
        return selected

    def reduce_cover(selected):
        if not selected:
            return selected
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1

        active = [True] * len(selected)
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                if not active[i]:
                    continue
                b_idx = selected[i]
                tids = block_triples_np[b_idx]
                if np.min(cover_count[tids]) > 1:
                    active[i] = False
                    cover_count[tids] -= 1
                    changed = True
        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search(selected, deadline):
        """Try to remove a block and re-cover the gap with fewer blocks."""
        if len(selected) < 2:
            return selected
        selected = list(selected)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples_np[b_idx]] += 1

        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each block (in random order)
            order = list(range(len(selected)))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline - 5:
                    break
                b_idx = selected[i]
                # Check if removing this block creates uncovered triples
                new_counts = cover_count - 1  # conceptual
                # Actually compute
                cover_count[block_triples_np[b_idx]] -= 1
                uncovered = cover_count == 0
                if not uncovered.any():
                    # Block was redundant
                    selected.pop(i)
                    improved = True
                    break
                # Try to re-cover with 1 block
                # Find a block that covers all uncovered triples
                # uncovered count is small hopefully
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) <= 35:  # at most one block's worth
                    # Find block containing all these triples
                    # For each triple in unc_idx, the covering blocks are triple_blocks_np[t]
                    # Intersection
                    candidate_sets = [set(triple_blocks_np[t].tolist()) for t in unc_idx]
                    candidates = set.intersection(*candidate_sets) if candidate_sets else set()
                    # Exclude already-selected blocks
                    already = set(selected)
                    candidates -= already
                    if candidates:
                        # Pick one
                        new_b = next(iter(candidates))
                        selected.pop(i)  # remove the one we tried
                        selected.append(new_b)
                        # Update cover_count
                        cover_count[block_triples_np[new_b]] += 1
                        improved = True
                        break
                    else:
                        # Revert
                        cover_count[block_triples_np[b_idx]] += 1
                else:
                    # Too many uncovered to fix with one block, revert
                    cover_count[block_triples_np[b_idx]] += 1

        return reduce_cover(selected)

    best_cover = None

    # Phase 1: Multiple greedy restarts
    seed = 0
    while time.time() < deadline - 20:
        budget = deadline - 15
        cover = greedy_cover(seed, budget)
        cover = complete_cover(cover, deadline - 10)
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover

        seed += 1

    if best_cover is None:
        best_cover = greedy_cover(0, deadline - 5)
        best_cover = complete_cover(best_cover, deadline - 2)
        best_cover = reduce_cover(best_cover)

    # Phase 2: Local search on the best cover
    best_cover = local_search(best_cover, deadline - 3)

    # Phase 3: If time remains, try more restarts targeting smaller covers
    while time.time() < deadline - 3:
        seed = random.randint(1000, 100000)
        budget = deadline - 5
        cover = greedy_cover(seed, budget)
        cover = complete_cover(cover, deadline - 3)
        cover = reduce_cover(cover)
        if len(cover) < len(best_cover):
            best_cover = cover

    # Final safety: ensure validity
    cover_count = np.zeros(num_triples, dtype=np.int32)
    for b_idx in best_cover:
        cover_count[block_triples_np[b_idx]] += 1
    if (cover_count == 0).any():
        # Fill in any gaps
        best_cover = complete_cover(best_cover, deadline)

    return [list(all_k_blocks[i]) for i in best_cover]
