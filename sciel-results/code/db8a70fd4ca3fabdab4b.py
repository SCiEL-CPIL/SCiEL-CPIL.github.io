import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}

    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Precompute block_indices: for each block, the indices of its t-subsets
    t_subsets_per_block = int(math.comb(k, t))
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)

    for i, b in enumerate(all_blocks):
        block_indices[i] = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]

    # Precompute frequency of each triple across all blocks
    # freq[j] = number of blocks that cover triple j
    freq = np.bincount(block_indices.flatten(), minlength=num_triples)

    # Inverse frequency for weighting (higher weight for rare triples)
    inv_freq = 1.0 / (freq + 1e-9)

    best_sol = None
    best_energy = float('inf')

    seed_counter = 0

    while time.time() < deadline - 5:
        seed_counter += 1
        rng = random.Random(seed_counter)

        # Choose strategy: 0=greedy, 1=weighted, 2=randomized
        strategy = seed_counter % 3

        # 1. Greedy Construction
        if strategy == 0:
            sol = _greedy_cover_vectorized(num_blocks, block_indices, num_triples, mode='greedy', deadline=deadline)
        elif strategy == 1:
            sol = _greedy_cover_vectorized(num_blocks, block_indices, num_triples, mode='weighted', inv_freq=inv_freq, deadline=deadline)
        else:
            sol = _greedy_cover_vectorized(num_blocks, block_indices, num_triples, mode='randomized', inv_freq=inv_freq, rng=rng, deadline=deadline)

        if sol is None:
            continue

        # 2. Remove Redundant Blocks
        sol = _remove_redundant(sol, block_indices, num_triples)

        # 3. Local Search
        time_left_for_sa = deadline - 5 - time.time()
        if time_left_for_sa < 1:
            break

        sol = _local_search(sol, num_blocks, block_indices, num_triples, time_left_for_sa, rng)

        # Evaluate
        if sol:
            # Check validity
            covered = np.zeros(num_triples, dtype=bool)
            for b_idx in sol:
                covered[block_indices[b_idx]] = True

            uncovered_count = np.sum(~covered)

            if uncovered_count == 0:
                energy = len(sol)
                if energy < best_energy:
                    best_energy = energy
                    best_sol = list(sol)

    if best_sol is None:
        return []

    return [list(all_blocks[i]) for i in best_sol]

def _greedy_cover_vectorized(num_blocks, block_indices, num_triples, mode='greedy', inv_freq=None, rng=None, deadline=None):
    """
    Vectorized greedy set cover.
    mode: 'greedy' (max new coverage), 'weighted' (max weighted new coverage), 'randomized' (random from top candidates)
    """
    uncovered = np.ones(num_triples, dtype=bool)
    selected_indices = []
    selected_set = set()

    while np.any(uncovered):
        if deadline is not None and time.time() > deadline:
            return None

        if mode == 'greedy':
            # Gain = number of uncovered triples covered by the block
            gains = uncovered[block_indices].sum(axis=1)
            # Exclude already selected
            if selected_set:
                gains[list(selected_set)] = -1
            best_gain = np.max(gains)
            if best_gain <= 0:
                break
            best_idx = int(np.argmax(gains))
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            uncovered[block_indices[best_idx]] = False

        elif mode == 'weighted':
            # Weighted gain = sum of inv_freq for uncovered triples
            w = np.where(uncovered, inv_freq, 0.0)
            gains = w[block_indices].sum(axis=1)
            if selected_set:
                gains[list(selected_set)] = -1
            best_gain = np.max(gains)
            if best_gain <= 0:
                break
            best_idx = int(np.argmax(gains))
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            uncovered[block_indices[best_idx]] = False

        elif mode == 'randomized':
            # Calculate gains, then pick randomly from top candidates
            gains = uncovered[block_indices].sum(axis=1)
            if selected_set:
                gains[list(selected_set)] = -1
            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            # Get indices of blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            # If too many candidates, sample a few
            if len(candidates) > 10:
                candidates = rng.choice(candidates, size=10, replace=False)

            best_idx = int(rng.choice(candidates))
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            uncovered[block_indices[best_idx]] = False

        else:
            raise ValueError("Unknown mode")

    if np.any(uncovered):
        return None

    return selected_indices

def _remove_redundant(sol, block_indices, num_triples):
    if not sol:
        return sol

    # Count coverage
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        counts[block_indices[idx]] += 1

    sol_list = list(sol)
    changed = True
    while changed:
        changed = False
        for i in range(len(sol_list)):
            idx = sol_list[i]
            # Check if removing idx leaves any triple with count 0
            # i.e., if any triple covered by idx has count 1
            if np.any(counts[block_indices[idx]] == 1):
                continue
            # Redundant
            counts[block_indices[idx]] -= 1
            sol_list.pop(i)
            changed = True
            break  # Restart loop as indices might have shifted

    return sol_list

def _local_search(initial_sol, num_blocks, block_indices, num_triples, time_limit, rng):
    sol = list(initial_sol)
    sol_set = set(sol)

    # Coverage counts
    counts = np.zeros(num_triples, dtype=np.int32)
    for idx in sol:
        counts[block_indices[idx]] += 1

    current_uncovered = np.sum(counts == 0)
    penalty = 100
    current_energy = len(sol) + penalty * current_uncovered
    best_sol = list(sol)
    best_energy = current_energy

    T = 5.0
    cooling = 0.99

    # Precompute list of all block indices for random selection
    all_block_indices = list(range(num_blocks))

    while time.time() < time_limit and T > 0.01:
        # Move: Remove one block, then optionally add one block to repair
        if not sol:
            break

        to_remove = rng.choice(sol)

        # Identify triples that will become uncovered
        block_mask = block_indices[to_remove]
        # Triples with count 1 in this block will become 0
        newly_uncovered_mask = np.zeros(num_triples, dtype=bool)
        newly_uncovered_mask[block_mask] = (counts[block_mask] == 1)

        # Tentatively remove
        counts[block_mask] -= 1
        sol.remove(to_remove)
        sol_set.discard(to_remove)

        new_uncovered = np.sum(counts == 0)

        if new_uncovered == 0:
            # Pure removal, always accept
            current_energy = len(sol)
            if current_energy < best_energy:
                best_energy = current_energy
                best_sol = list(sol)
        else:
            # Need to add a block to cover newly_uncovered
            # Find block that covers most of newly_uncovered
            gains = newly_uncovered_mask[block_indices].sum(axis=1)

            # Mask out selected blocks
            gains[list(sol_set)] = -1

            max_gain = np.max(gains)

            if max_gain > 0:
                candidates = np.where(gains == max_gain)[0]
                to_add = int(rng.choice(candidates))

                # Add the block
                counts[block_indices[to_add]] += 1
                sol.append(to_add)
                sol_set.add(to_add)

                # Recalculate energy
                final_uncovered = np.sum(counts == 0)
                final_energy = len(sol) + penalty * final_uncovered

                delta_e = final_energy - current_energy

                if delta_e <= 0 or rng.random() < np.exp(-delta_e / T):
                    current_energy = final_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                else:
                    # Reject: Undo
                    counts[block_indices[to_add]] -= 1
                    sol.remove(to_add)
                    sol_set.discard(to_add)
                    counts[block_mask] += 1  # Restore removed block
                    sol.append(to_remove)
                    sol_set.add(to_remove)
                    # current_energy remains unchanged
            else:
                # No block can cover the newly uncovered triples?
                # This shouldn't happen if the cover was valid initially,
                # because the removed block itself covered them, but it's not available.
                # Wait, if newly_uncovered_mask is not empty, there must be blocks covering them.
                # But if all blocks covering them are already in sol?
                # Then we can't repair. We must undo the removal.
                counts[block_mask] += 1
                sol.append(to_remove)
                sol_set.add(to_remove)
                # current_energy remains unchanged

        T *= cooling

    return best_sol
