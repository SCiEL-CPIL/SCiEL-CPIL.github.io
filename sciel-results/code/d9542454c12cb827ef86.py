import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks and their covered triplets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    block_triplet_ids = []
    for block in all_blocks:
        tids = [triplet_to_id[triplet] for triplet in itertools.combinations(block, t)]
        block_triplet_ids.append(tids)

    # Reverse index: triplet -> list of block indices covering it
    triplet_blocks = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triplet_ids):
        for tid in tids:
            triplet_blocks[tid].append(b_idx)

    def greedy_block_construct(seed, top_k=1):
        """Greedy over all k-blocks with random tie-breaking."""
        rng = random.Random(seed)
        gain = np.array([len(ts) for ts in block_triplet_ids], dtype=np.int32)
        is_uncovered = np.ones(n_triplets, dtype=np.bool_)
        selected = []

        while is_uncovered.any():
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            # Find candidates with max gain
            candidates = np.where(gain == max_gain)[0]
            if top_k > 1 and len(candidates) > top_k:
                # Pick randomly from top-k fraction
                n_pick = min(top_k, len(candidates))
                idxs = rng.sample(candidates.tolist(), n_pick)
                b_idx = idxs[rng.randrange(n_pick)]
            else:
                b_idx = int(rng.choice(candidates))

            selected.append(b_idx)

            # Update: for each triplet in this block that was uncovered,
            # mark as covered and decrement gain for all blocks covering it
            for tid in block_triplet_ids[b_idx]:
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    for other_b in triplet_blocks[tid]:
                        gain[other_b] -= 1

        return selected

    def build_cover(selected_indices):
        """Convert block indices to list of blocks (sorted lists of points)."""
        return [list(all_blocks[i]) for i in selected_indices]

    def reduce_and_optimize(blocks, time_budget=5.0):
        """Remove redundant blocks and try swaps to reduce count."""
        count = np.zeros(n_triplets, dtype=np.int32)

        def update_counts(blks, delta=1):
            for blk in blks:
                for triplet in itertools.combinations(blk, t):
                    tid = triplet_to_id[triplet]
                    count[tid] += delta

        update_counts(blocks, 1)
        current = list(blocks)

        def is_redundant(block):
            for triplet in itertools.combinations(block, t):
                if count[triplet_to_id[triplet]] == 1:
                    return False
            return True

        def try_remove_redundant(blks):
            removed = False
            new_blks = []
            for b in blks:
                if is_redundant(b):
                    update_counts([b], -1)
                    removed = True
                else:
                    new_blks.append(b)
            return new_blks, removed

        def try_swap_2_for_1(blks):
            n = len(blks)
            if n < 2:
                return blks, False

            # Precompute triplet id sets for each block
            block_tids = []
            for b in blks:
                tids = set()
                for triplet in itertools.combinations(b, t):
                    tids.add(triplet_to_id[triplet])
                block_tids.append(tids)

            for i in range(n):
                if time.time() > deadline - 2:
                    return blks, False
                b1 = blks[i]
                ids_b1 = block_tids[i]

                for j in range(i + 1, n):
                    if time.time() > deadline - 2:
                        return blks, False
                    b2 = blks[j]
                    ids_b2 = block_tids[j]

                    critical_points = set()
                    for tid in ids_b1:
                        c = count[tid]
                        in_b2 = 1 if tid in ids_b2 else 0
                        if c == 1 + in_b2:
                            critical_points.update(all_triplets[tid])

                    for tid in ids_b2:
                        if tid in ids_b1:
                            continue
                        c = count[tid]
                        if c == 1:
                            critical_points.update(all_triplets[tid])

                    if not critical_points:
                        # Both redundant together
                        update_counts([b1, b2], -1)
                        return [b for idx, b in enumerate(blks) if idx != i and idx != j], True

                    if len(critical_points) <= k:
                        new_block = sorted(critical_points)
                        fillers = [p for p in range(v) if p not in new_block]
                        rng_fill = random.Random(hash((i, j)) & 0xFFFF)
                        while len(new_block) < k:
                            p = rng_fill.choice(fillers)
                            if p not in new_block:
                                new_block.append(p)
                        new_block.sort()

                        update_counts([b1, b2], -1)
                        update_counts([new_block], 1)

                        new_blks = [b for idx, b in enumerate(blks) if idx != i and idx != j]
                        new_blks.append(new_block)
                        return new_blks, True

            return blks, False

        def try_swap_1_for_1(blks):
            """Try replacing one block with another to maintain cover while allowing further reduction."""
            # This is harder; skip for now as 2-for-1 is more impactful
            return blks, False

        progress = True
        while progress and time.time() < deadline - 2:
            progress = False
            current, removed = try_remove_redundant(current)
            if removed:
                progress = True
            if not progress:
                current, swapped = try_swap_2_for_1(current)
                if swapped:
                    progress = True

        current, _ = try_remove_redundant(current)
        return current

    # Phase 1: Multiple greedy constructions with different seeds
    best_blocks = None
    best_count = float('inf')

    # Use varied top_k for diversity
    for seed in range(30):
        if time.time() > deadline - 40:
            break
        top_k = 1 if seed < 15 else 3
        selected = greedy_block_construct(seed, top_k=top_k)
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = build_cover(selected)

    if best_blocks is None:
        selected = greedy_block_construct(0)
        best_blocks = build_cover(selected)
        best_count = len(selected)

    # Phase 2: Local search optimization on the best cover
    if time.time() < deadline - 3:
        optimized = reduce_and_optimize(best_blocks)
        if len(optimized) < best_count:
            best_blocks = optimized
            best_count = len(optimized)

    # Phase 3: If we have time, try optimizing second-best cover too
    # (skip for now to keep it simple)

    return best_blocks


if __name__ == "__main__":
    pass
