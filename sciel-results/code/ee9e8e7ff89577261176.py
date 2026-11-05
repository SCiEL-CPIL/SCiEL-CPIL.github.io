import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(12))
    all_blocks = list(itertools.combinations(points, 6))
    all_5_subsets = list(itertools.combinations(points, 5))
    num_blocks = len(all_blocks)
    num_constraints = len(all_5_subsets)

    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}

    # Pre-compute 5-subset indices for each block
    block_5_indices = []
    for b in all_blocks:
        indices = [subset_to_idx[s] for s in itertools.combinations(b, 5)]
        block_5_indices.append(indices)

    # Build conflict graph: for each block, which other blocks share a 5-subset
    # This is expensive to compute fully, so we'll compute on-the-fly or use a smarter approach
    # For Algorithm X / Dancing Links, we need a different formulation

    # Actually, let's use a direct construction approach for the Witt design S(5,6,12)
    # The Steiner system S(5,6,12) can be constructed using the Mathieu group M12

    # Direct construction of S(5,6,12) blocks:
    # The Witt design S(5,6,12) has 132 blocks. We can construct it using the following method:
    # 1. Start with the 12 points {0, 1, ..., 11}
    # 2. The design is highly symmetric - it's the unique Steiner system S(5,6,12)

    # Let's use a known explicit construction. The blocks of S(5,6,12) can be generated
    # from the Steiner system S(4,5,11) on 11 points plus one extra point.

    # Alternative: Use the fact that S(5,6,12) is the complement of S(5,6,12) in some sense,
    # or construct it via the Mathieu group.

    # Let's try a recursive exact approach using backtracking with pruning:
    # We want to find a set of 132 blocks such that no 5-subset appears in two blocks.
    # This is equivalent to finding a maximum packing.

    # For efficiency, let's use a greedy approach with restarts and local search,
    # but with a much better initial ordering.

    # Actually, let's try to directly construct the Witt design using its known structure.
    # The 132 blocks of S(5,6,12) can be partitioned into:
    # - 11 "ovoids" of 12 blocks each (related to S(4,5,11))
    # - Plus additional blocks

    # Let me use a different approach: enumerate all 924 blocks and use a maximum
    # independent set approach on the conflict graph, or use integer programming.

    # Given the time constraints, let's use a highly optimized greedy with many restarts
    # and a better heuristic for block ordering.

    # Precompute for each 5-subset which blocks contain it
    constraint_to_blocks = [[] for _ in range(num_constraints)]
    for j, indices in enumerate(block_5_indices):
        for i in indices:
            constraint_to_blocks[i].append(j)

    def run_greedy(seed, ordering_func=None):
        rng = np.random.RandomState(seed)
        covered = np.zeros(num_constraints, dtype=bool)
        picked = []
        valid_blocks = np.ones(num_blocks, dtype=bool)

        if ordering_func is not None:
            order = ordering_func(num_blocks, valid_blocks, covered, constraint_to_blocks)
        else:
            order = rng.permutation(num_blocks)

        for j in order:
            if not valid_blocks[j]:
                continue
            indices = block_5_indices[j]
            if not np.any(covered[indices]):
                picked.append(list(all_blocks[j]))
                covered[indices] = True
                # Invalidate all blocks that share any 5-subset with this block
                for i in indices:
                    for k in constraint_to_blocks[i]:
                        valid_blocks[k] = False

        return picked

    best_blocks = []
    num_restarts = 2000

    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # Try different ordering strategies
        if i % 3 == 0:
            # Random order
            blocks = run_greedy(i)
        elif i % 3 == 1:
            # Order by "density" - prefer blocks that conflict with fewer other blocks
            def density_order(n, valid, cov, c2b):
                # For each valid block, count how many other valid blocks it conflicts with
                # Lower conflict count = better to pick first
                scores = np.zeros(n)
                for j in range(n):
                    if not valid[j]:
                        scores[j] = -1
                        continue
                    conflict_count = 0
                    for i5 in block_5_indices[j]:
                        for k in c2b[i5]:
                            if valid[k] and k != j:
                                conflict_count += 1
                    scores[j] = conflict_count
                # Pick in order of increasing conflict count (with some randomness)
                order = np.argsort(scores)
                return order

            blocks = run_greedy(i, density_order)
        else:
            # Order by "rarity" - prefer blocks containing 5-subsets that appear in fewer blocks
            def rarity_order(n, valid, cov, c2b):
                scores = np.zeros(n)
                for j in range(n):
                    if not valid[j]:
                        scores[j] = -1
                        continue
                    rarity = sum(len(c2b[i5]) for i5 in block_5_indices[j])
                    scores[j] = rarity
                order = np.argsort(scores)
                return order

            blocks = run_greedy(i, rarity_order)

        if len(blocks) > len(best_blocks):
            best_blocks = blocks
            if len(best_blocks) == 132:
                break

    return best_blocks
