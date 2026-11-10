"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 110.0
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_size = 9 * (3 ** (n - 4))
    base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    if len(base) <= 156:
        return base

    total_words = q ** n
    if total_words > 500000:
        return base

    # Precompute all words as indices
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Convert base to array
    base_arr = np.array(base, dtype=np.uint8)  # (base_size, n)

    # Compute covering sets as boolean array (base_size, total_words)
    # For n=7, base_size=2187, total_words=2187: 2187*2187 ≈ 4.8M booleans, feasible
    if base_size * total_words > 200_000_000:
        return base

    # Compute Hamming distances in chunks
    chunk_size = max(1, 200_000_000 // (total_words * n))
    covering_sets = []
    for i in range(0, base_size, chunk_size):
        chunk = base_arr[i:i + chunk_size]
        dist = np.sum(chunk[:, np.newaxis, :] != all_words[np.newaxis, :, :], axis=2)
        covered = dist <= radius
        covering_sets.append(covered)

    covering_sets = np.concatenate(covering_sets, axis=0)  # (base_size, total_words)

    # Phase 1: Greedy set cover to get an initial solution
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected = []
    selected_mask = np.zeros(base_size, dtype=np.uint8)

    # Start with a random codeword
    rng = random.Random(42)
    idx = rng.randint(0, base_size - 1)
    selected.append(idx)
    selected_mask[idx] = 1
    uncovered[covering_sets[idx]] = 0

    while uncovered.sum() > 0 and time.time() < deadline - 60:
        gains = (covering_sets * uncovered).sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        selected.append(best)
        selected_mask[best] = 1
        uncovered[covering_sets[best]] = 0

    # Phase 2: Simulated annealing to reduce the number of codewords
    # Energy function: primarily penalize uncovered words, secondarily count selected codewords
    # E = M * uncovered_count + selected_count
    # where M is a large penalty for uncovered words

    M = 100  # Penalty multiplier for uncovered words

    def compute_energy(selected_set, uncovered_arr):
        return M * int(uncovered_arr.sum()) + len(selected_set)

    current_selected = list(selected)
    current_uncovered = uncovered.copy()
    current_energy = compute_energy(current_selected, current_uncovered)
    best_selected = list(current_selected)
    best_uncovered = current_uncovered.copy()
    best_energy = current_energy

    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iter = 100000

    T = T_start
    iter_count = 0

    while T > T_end and iter_count < max_iter and time.time() < deadline - 5:
        iter_count += 1

        # Try to remove a random selected codeword
        if len(current_selected) > 1:
            remove_idx = rng.randint(0, len(current_selected) - 1)
            codeword_to_remove = current_selected[remove_idx]

            # Compute what would be uncovered if we remove this codeword
            # Words that were only covered by this codeword
            # A word is newly uncovered if:
            # 1. It was covered by this codeword
            # 2. It is not covered by any other selected codeword

            # Check if removing this codeword creates new uncovered words
            # Count how many words are covered ONLY by this codeword among selected
            # For efficiency, check a subset or use the covering matrix

            # Simple approach: temporarily remove and check
            new_uncovered = current_uncovered.copy()
            # Words covered by the removed codeword that become uncovered
            # A word w is newly uncovered if covering_sets[codeword_to_remove][w] == 1
            # AND no other selected codeword covers w

            # This is expensive to check exactly, so use a heuristic:
            # Only try removal if the codeword covers few unique words
            # Approximate by checking if the codeword's coverage overlaps heavily with others

            # For now, just try the removal and compute the actual change
            # Remove from current_selected
            current_selected.pop(remove_idx)

            # Recompute uncovered
            # A word is uncovered if no selected codeword covers it
            # Efficient: use the covering_sets matrix
            # uncovered[w] = 1 if sum(covering_sets[selected, w] for selected in current_selected) == 0

            if len(current_selected) == 0:
                new_uncovered = np.ones(total_words, dtype=np.uint8)
            else:
                sel_arr = np.array(current_selected)
                # Sum coverage over selected codewords
                coverage = covering_sets[sel_arr].sum(axis=0)
                new_uncovered = (coverage == 0).astype(np.uint8)

            new_energy = compute_energy(current_selected, new_uncovered)
            delta_E = new_energy - current_energy

            # Accept with probability exp(-delta_E / T)
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept the move
                current_uncovered = new_uncovered
                current_energy = new_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = list(current_selected)
                    best_uncovered = current_uncovered.copy()
            else:
                # Reject the move, put the codeword back
                current_selected.insert(remove_idx, codeword_to_remove)

        # Try to swap: remove one and add another
        elif len(current_selected) > 1:
            remove_idx = rng.randint(0, len(current_selected) - 1)
            codeword_to_remove = current_selected[remove_idx]

            # Pick a random codeword to add (not currently selected)
            # For efficiency, pick randomly from all codewords
            codeword_to_add = rng.randint(0, base_size - 1)

            if codeword_to_add not in current_selected:
                # Perform the swap
                current_selected[remove_idx] = codeword_to_add

                # Recompute uncovered
                if len(current_selected) == 0:
                    new_uncovered = np.ones(total_words, dtype=np.uint8)
                else:
                    sel_arr = np.array(current_selected)
                    coverage = covering_sets[sel_arr].sum(axis=0)
                    new_uncovered = (coverage == 0).astype(np.uint8)

                new_energy = compute_energy(current_selected, new_uncovered)
                delta_E = new_energy - current_energy

                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    current_uncovered = new_uncovered
                    current_energy = new_energy

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_selected = list(current_selected)
                        best_uncovered = current_uncovered.copy()
                else:
                    # Reject, restore
                    current_selected[remove_idx] = codeword_to_remove

        # Cool down
        T *= cooling_rate

    # Return the best solution found
    result = [base[i] for i in best_selected]
    return result
# EVOLVE-BLOCK-END
