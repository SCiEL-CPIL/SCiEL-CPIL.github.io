import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    num_triples = len(all_t_subsets)

    # Precompute: for each pair (a,b), array of triple indices containing both a and b
    # For t=3, each pair appears in exactly v-2 triples (one per possible third point)
    # pair_triple_map[(a,b)] = array of length v-2, where entry j is the triple index
    #   for the triple (a, b, point_j) where point_j is the j-th point not in {a,b}
    # We also need a way to quickly find the triple index for (a,b,x)
    # triple_idx_fast[a][b][x] = index of sorted triple (a,b,x)
    
    # For t=3: precompute a 3D array triple_idx3[a][b][c] = index of (a,b,c) sorted
    # But that's 24^3 = 13824 entries, manageable
    if t == 3:
        # triple_idx3[a][b][c] = triple_to_idx[tuple(sorted((a,b,c)))]
        triple_idx3 = np.full((v, v, v), -1, dtype=np.int32)
        for a in range(v):
            for b in range(v):
                if a == b:
                    continue
                for c in range(v):
                    if c == a or c == b:
                        continue
                    tr = tuple(sorted((a, b, c)))
                    triple_idx3[a, b, c] = triple_to_idx[tr]
    
    def block_triple_indices(block):
        """Return list of triple indices covered by this block."""
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[tuple(triple)])
        return indices

    def random_greedy_cover(rng):
        """Fast greedy with vectorized gain computation. Returns list of blocks."""
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []

        while True:
            if time.time() > deadline - 5.0:
                break
            if not np.any(uncovered):
                break

            # Pick a random uncovered triple as seed
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = rng.choice(uncovered_indices.tolist())
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)

            # Extend block to size k
            while len(block) < k:
                if time.time() > deadline - 5.0:
                    break

                candidates = [x for x in all_points if x not in block_set]
                if not candidates:
                    break

                # Compute gain for each candidate using vectorized triple index lookup
                # For t=3: gain(x) = sum over pairs (a,b) in block of uncovered[idx(x,a,b)]
                block_pairs = list(itertools.combinations(block, t - 1))
                
                if t == 3 and len(block_pairs) > 0 and len(candidates) > 0:
                    # Vectorized: for each pair (a,b) and each candidate x,
                    # get triple index from triple_idx3[a][b][x]
                    n_pairs = len(block_pairs)
                    n_cands = len(candidates)
                    # Build index array: shape (n_pairs, n_cands)
                    idx_array = np.zeros((n_pairs, n_cands), dtype=np.int32)
                    for pi, (a, b) in enumerate(block_pairs):
                        for ci, x in enumerate(candidates):
                            idx_array[pi, ci] = triple_idx3[a, b, x]
                    # gains = sum over pairs of uncovered[idx]
                    gains = uncovered[idx_array].sum(axis=0)
                else:
                    # Fallback for general t
                    gains = np.zeros(len(candidates), dtype=np.int32)
                    for x_i, x in enumerate(candidates):
                        for pair in block_pairs:
                            triple = tuple(sorted(pair + (x,)))
                            idx = triple_to_idx[triple]
                            gains[x_i] += uncovered[idx]

                # Pick candidate with highest gain, break ties randomly
                max_gain = int(np.max(gains))
                if max_gain <= 0:
                    block.append(rng.choice(candidates))
                else:
                    best_indices = np.where(gains == max_gain)[0]
                    chosen = rng.choice(best_indices.tolist())
                    block.append(candidates[chosen])

                block_set.add(block[-1])

            # Fill remaining if needed
            if len(block) < k:
                block_set = set(block)
                while len(block) < k:
                    available = [x for x in all_points if x not in block_set]
                    if not available:
                        break
                    block.append(rng.choice(available))
                    block_set.add(block[-1])

            block.sort()
            blocks.append(block)

            # Update uncovered
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0

        return blocks

    def local_search_remove_redundant(blocks):
        """Remove redundant blocks. Returns new list of blocks."""
        if not blocks:
            return blocks

        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []

        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1

        changed = True
        while changed:
            changed = False
            to_remove = []

            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    to_remove.append(i)

            if to_remove:
                remove_set = set(to_remove)
                new_blocks = []
                new_block_indices = []
                for i in range(len(blocks)):
                    if i not in remove_set:
                        new_blocks.append(blocks[i])
                        new_block_indices.append(block_indices[i])

                for i in to_remove:
                    triple_count[block_indices[i]] -= 1

                blocks = new_blocks
                block_indices = new_block_indices
                changed = True

        return blocks

    def local_search_swap(blocks):
        """Try to swap points in blocks to enable further deletions. Returns improved blocks."""
        if not blocks:
            return blocks

        triple_count = np.zeros(num_triples, dtype=np.int32)
        block_indices = []

        for block in blocks:
            indices = block_triple_indices(block)
            block_indices.append(indices)
            triple_count[indices] += 1

        improved = False
        max_iterations = 200

        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break

            # Try to delete blocks first
            deleted = False
            for i in range(len(blocks)):
                is_redundant = True
                for idx in block_indices[i]:
                    if triple_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    triple_count[block_indices[i]] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    deleted = True
                    improved = True
                    break

            if deleted:
                continue

            # Try swaps: pick a random block, swap one point with another
            if len(blocks) == 0:
                break

            found_swap = False
            for _ in range(30):  # Try up to 30 random swaps per iteration
                if time.time() > deadline - 5.0:
                    break

                idx = random.randint(0, len(blocks) - 1)
                block = blocks[idx]
                old_x = random.choice(block)
                new_x = random.randint(0, v - 1)

                if new_x == old_x or new_x in block:
                    continue

                # Compute new block
                other_points = [p for p in block if p != old_x]
                new_block = sorted(other_points + [new_x])

                old_triples = set(block_triple_indices(block))
                new_triples = set(block_triple_indices(new_block))

                # Check: all triples only in old_block must have count > 1
                lost = old_triples - new_triples
                valid = True
                for tri_idx in lost:
                    if triple_count[tri_idx] <= 1:
                        valid = False
                        break

                if valid:
                    # Apply swap
                    triple_count[list(old_triples)] -= 1
                    triple_count[list(new_triples)] += 1
                    blocks[idx] = new_block
                    block_indices[idx] = block_triple_indices(new_block)
                    found_swap = True
                    improved = True
                    break

            if not found_swap:
                # No swap found in this round; stop to avoid infinite loop
                break

        return blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')

    # Phase 1: Greedy restarts (60% of time)
    greedy_deadline = start_time + (deadline - start_time) * 0.60
    seed = 42

    while time.time() < greedy_deadline:
        rng = random.Random(seed)
        seed += 1
        blocks = random_greedy_cover(rng)
        if not blocks:
            continue

        # Quick redundancy removal
        blocks = local_search_remove_redundant(blocks)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]

    # Phase 2: Local search refinement on best solution (35% of time)
    if best_blocks is not None and time.time() < deadline - 5.0:
        refine_deadline = deadline - 5.0
        current_blocks = [b[:] for b in best_blocks]
        current_len = len(current_blocks)

        # Run swap-based local search multiple times with different random seeds
        for attempt in range(20):
            if time.time() > refine_deadline:
                break

            trial_blocks = [b[:] for b in current_blocks]
            trial_blocks = local_search_swap(trial_blocks)

            if len(trial_blocks) < current_len:
                current_blocks = trial_blocks
                current_len = len(trial_blocks)
                if current_len < best_len:
                    best_len = current_len
                    best_blocks = [b[:] for b in current_blocks]
            else:
                # Perturb: try a fresh greedy + merge
                if time.time() > refine_deadline:
                    break
                rng = random.Random(seed)
                seed += 1
                greedy_blocks = random_greedy_cover(rng)
                if greedy_blocks:
                    greedy_blocks = local_search_remove_redundant(greedy_blocks)
                    if len(greedy_blocks) < current_len:
                        current_blocks = greedy_blocks
                        current_len = len(greedy_blocks)
                        if current_len < best_len:
                            best_len = current_len
                            best_blocks = [b[:] for b in current_blocks]

    # Final: ensure validity
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(num_triples, dtype=np.uint8)
        blocks = []
        while np.any(uncovered) and time.time() < deadline - 2.0:
            uncovered_indices = np.where(uncovered)[0]
            seed_idx = uncovered_indices[0]
            seed_triple = all_t_subsets[seed_idx]
            block = list(seed_triple)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(pair + (x,)))
                        idx = triple_to_idx[triple]
                        gain += uncovered[idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
            if len(block) < k:
                available = [x for x in all_points if x not in block_set]
                while len(block) < k and available:
                    block.append(available.pop())
            block.sort()
            blocks.append(block)
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[tuple(triple)]
                uncovered[idx] = 0
        best_blocks = blocks
        best_len = len(blocks)

    return best_blocks
