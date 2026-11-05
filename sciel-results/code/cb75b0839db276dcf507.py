import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks and their triplet memberships
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)

    # block_triple_indices[i, j] = triplet_id of the j-th t-subset of block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triplet_to_id[sub]

    def greedy_cover(random_seed, bias=0.0):
        """Greedy set cover with optional random tie-breaking bias."""
        rng = random.Random(random_seed)
        uncovered = np.ones(n_triplets, dtype=bool)
        selected = []
        remaining = n_triplets

        while remaining > 0:
            # Compute gains for all blocks
            gains = np.sum(uncovered[block_triple_indices], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            if bias > 0 and rng.random() < bias and len(candidates) > 1:
                # Pick a random candidate (not necessarily max gain)
                best_idx = candidates[rng.randint(len(candidates))]
            else:
                # Pick random among max-gain candidates
                best_idx = candidates[rng.randint(len(candidates))]

            block_idx = block_triple_indices[best_idx]
            newly_covered = np.sum(uncovered[block_idx])
            uncovered[block_idx] = False
            remaining -= newly_covered
            selected.append(list(all_blocks[best_idx]))

        return selected

    def simulate_annealing(initial_blocks, time_limit, seed=0):
        """
        Simulated annealing for set cover optimization.
        Moves:
          1. Remove a redundant block (if all its triplets are covered by others)
          2. Swap: remove one block, add another (may temporarily increase uncovered count)
          3. 2-for-1: remove two blocks, add one new block
        Acceptance: Metropolis criterion based on change in block count.
        """
        rng = np.random.RandomState(seed)
        current_blocks = list(initial_blocks)
        best_blocks = list(initial_blocks)
        best_count = len(current_blocks)

        # Maintain coverage counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in current_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        def is_redundant(block):
            for triplet in itertools.combinations(block, t):
                if counts[triplet_to_id[triplet]] == 1:
                    return False
            return True

        def remove_block(block):
            for triplet in itertools.combinations(block, t):
                counts[triplet_to_id[triplet]] -= 1

        def add_block(block):
            for triplet in itertools.combinations(block, t):
                counts[triplet_to_id[triplet]] += 1

        # Temperature schedule
        T0 = 2.0
        T_end = 0.01
        total_steps = 5000
        cooling = (T_end / T0) ** (1.0 / total_steps)

        for step in range(total_steps):
            if time.time() > time_limit:
                break
            T = T0 * (cooling ** step)

            # Choose a move
            move_type = rng.randint(0, 2)
            n = len(current_blocks)
            if n == 0:
                break

            if move_type == 0 and n > 1:
                # Try removing a redundant block
                idx = rng.randint(0, n - 1)
                block = current_blocks[idx]
                if is_redundant(block):
                    remove_block(block)
                    current_blocks.pop(idx)
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
                # If not redundant, reject (no change)

            elif move_type == 1 and n > 0:
                # Swap: remove one block, add a random different block
                idx = rng.randint(0, n - 1)
                old_block = current_blocks[idx]
                new_block_idx = rng.randint(0, n_blocks - 1)
                new_block = list(all_blocks[new_block_idx])

                # Compute delta in block count: 0 (swap), but check validity
                # We need all triplets that were only covered by old_block to be covered by new_block
                # or by other blocks
                # Temporarily remove old_block
                remove_block(old_block)
                # Check if any triplet becomes uncovered
                uncovered_after = np.any(counts < 0)  # shouldn't happen
                # Actually check: are all triplets still covered?
                min_count = np.min(counts)
                if min_count < 0:
                    # This shouldn't happen with proper counting
                    add_block(old_block)
                    continue

                # Add new block
                add_block(new_block)
                current_blocks[idx] = new_block

                # Check if valid: all triplets covered
                if np.min(counts) >= 0:
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    # Invalid, revert
                    remove_block(new_block)
                    add_block(old_block)
                    current_blocks[idx] = old_block

            elif move_type == 2 and n >= 2:
                # 2-for-1: remove two blocks, add one
                i, j = rng.choice(np.arange(n), size=2, replace=False)
                i, j = int(i), int(j)
                if i == j:
                    continue
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                # Find critical triplets: those covered only by b1 or b2
                critical = set()
                for triplet in itertools.combinations(b1, t):
                    tid = triplet_to_id[triplet]
                    c = counts[tid]
                    covered_by_b2 = 1 if any(
                        triplet_to_id[sub] == tid for sub in itertools.combinations(b2, t)
                    ) else 0
                    if c == 1 + covered_by_b2:
                        critical.add(tuple(triplet))

                for triplet in itertools.combinations(b2, t):
                    tid = triplet_to_id[triplet]
                    c = counts[tid]
                    if c == 1:
                        critical.add(tuple(triplet))

                if len(critical) > k:
                    continue

                # Build new block containing all critical points
                new_block = list(set(p for trip in critical for p in trip))
                remaining_points = [p for p in range(v) if p not in new_block]
                need = k - len(new_block)
                if need > 0:
                    new_block.extend(rng.choice(remaining_points, size=min(need, len(remaining_points)), replace=False).tolist())
                new_block = sorted(new_block)

                # Remove b1 and b2
                remove_block(b1)
                remove_block(b2)
                # Add new block
                add_block(new_block)

                # Check validity
                if np.min(counts) >= 0:
                    current_blocks = [b for idx, b in enumerate(current_blocks) if idx not in (i, j)]
                    current_blocks.append(new_block)
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    # Revert
                    remove_block(new_block)
                    add_block(b1)
                    add_block(b2)
                    current_blocks[i] = b1
                    current_blocks[j] = b2

        return best_blocks, best_count

    def local_search_optimize(blocks, time_limit):
        """Deterministic local search: remove redundants, try 2-for-1 swaps."""
        current = list(blocks)
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in current:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1

        def is_redundant(block):
            for triplet in itertools.combinations(block, t):
                if counts[triplet_to_id[triplet]] == 1:
                    return False
            return True

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Remove redundants
            i = 0
            while i < len(current):
                if is_redundant(current[i]):
                    for triplet in itertools.combinations(current[i], t):
                        counts[triplet_to_id[triplet]] -= 1
                    current.pop(i)
                    improved = True
                else:
                    i += 1

            # Try 2-for-1
            n = len(current)
            if n >= 2:
                for i in range(n):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, n):
                        if time.time() > time_limit:
                            break
                        b1 = current[i]
                        b2 = current[j]
                        critical = set()
                        for triplet in itertools.combinations(b1, t):
                            tid = triplet_to_id[triplet]
                            c = counts[tid]
                            covered_by_b2 = 1 if any(
                                triplet_to_id[sub] == tid for sub in itertools.combinations(b2, t)
                            ) else 0
                            if c == 1 + covered_by_b2:
                                critical.add(tuple(triplet))
                        for triplet in itertools.combinations(b2, t):
                            tid = triplet_to_id[triplet]
                            if counts[tid] == 1:
                                critical.add(tuple(triplet))
                        if len(critical) <= k:
                            new_block = list(set(p for trip in critical for p in trip))
                            remaining_points = [p for p in range(v) if p not in new_block]
                            need = k - len(new_block)
                            if need > 0:
                                new_block.extend(remaining_points[:need])
                            new_block = sorted(new_block)
                            # Verify
                            for triplet in itertools.combinations(new_block, t):
                                pass
                            # Apply
                            for triplet in itertools.combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            for triplet in itertools.combinations(b2, t):
                                counts[triplet_to_id[triplet]] -= 1
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                            current = [b for idx, b in enumerate(current) if idx not in (i, j)]
                            current.append(new_block)
                            improved = True
                            break
                    if improved:
                        break

        return current

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    n_restarts = 30
    for seed in range(n_restarts):
        if time.time() > deadline - 40:
            break
        bias = 0.1 if seed % 3 == 0 else 0.0
        blocks = greedy_cover(seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_count = len(best_blocks)

    # Phase 2: Simulated Annealing from best greedy solution
    sa_time_limit = deadline - 10
    sa_blocks, sa_count = simulate_annealing(best_blocks, sa_time_limit, seed=42)
    if sa_count < best_count:
        best_count = sa_count
        best_blocks = sa_blocks

    # Phase 3: Local search refinement
    ls_time_limit = deadline - 2
    final_blocks = local_search_optimize(best_blocks, ls_time_limit)

    # Final redundancy removal
    counts = np.zeros(n_triplets, dtype=np.int32)
    for b in final_blocks:
        for triplet in itertools.combinations(b, t):
            counts[triplet_to_id[triplet]] += 1

    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if counts[triplet_to_id[triplet]] == 1:
                return False
        return True

    cleaned = []
    for b in final_blocks:
        if not is_redundant(b):
            cleaned.append(b)
        else:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] -= 1

    return cleaned

if __name__ == "__main__":
    pass
