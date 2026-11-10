"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # For q=3, n=7, radius=1
    total_words = q ** n  # 2187

    # Precompute neighbors for each word as an integer
    # neighbors[i] = list of integers within Hamming distance 1 of word i
    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result

    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word

    # Build neighbor lookup table
    neighbors_table = []
    for w_int in range(total_words):
        w = int_to_word(w_int)
        nbs = [w_int]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    w[i] = new_val
                    nbs.append(word_to_int(w))
            w[i] = orig
        neighbors_table.append(nbs)

    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    code = list(initial_code)
    code_ints = [word_to_int(c) for c in code]
    code_set = set(code_ints)

    # Simulated annealing over binary inclusion vector
    # State: selected[i] = True if word i is in the code
    # Energy: penalty * (number of uncovered words) + lambda * (number of selected words)

    # Precompute for each word, which other words it covers (i.e., neighbors)
    # coverage_matrix[i][j] = True if word i covers word j (i.e., dist(i,j) <= radius)
    # For radius=1, word i covers itself and all words at Hamming distance 1

    # We'll track covered_count[j] = number of selected words that cover word j
    covered_count = np.zeros(total_words, dtype=np.int32)
    selected = np.zeros(total_words, dtype=bool)

    # Initialize coverage
    for c_int in code_ints:
        selected[c_int] = True
        for nb in neighbors_table[c_int]:
            covered_count[nb] += 1

    def get_energy():
        uncovered = np.sum(covered_count == 0)
        selected_count = np.sum(selected)
        return uncovered * 1000 + selected_count

    current_energy = get_energy()
    best_energy = current_energy
    best_selected = selected.copy()

    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.9995

    # Annealing loop
    T = T_start
    step = 0
    while time.time() < deadline - 10 and T > T_end:
        # Pick a random word to toggle
        idx = random.randint(0, total_words - 1)

        if selected[idx]:
            # Try to remove
            nbs = neighbors_table[idx]
            # Check if any neighbor would become uncovered
            would_uncover = any(covered_count[nb] <= 1 for nb in nbs)

            if would_uncover:
                # Compute energy change: adding uncovered penalty
                delta_uncovered = sum(1 for nb in nbs if covered_count[nb] <= 1)
                delta_energy = delta_uncovered * 1000 - 1
            else:
                delta_energy = -1  # Remove one selected word, no new uncovered

            # Accept with probability exp(-delta_energy / T) if delta_energy > 0
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / T):
                # Apply change
                selected[idx] = False
                for nb in nbs:
                    covered_count[nb] -= 1
                new_energy = current_energy + delta_energy
                current_energy = new_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()

        else:
            # Try to add
            nbs = neighbors_table[idx]
            # Count how many currently uncovered words this would cover
            covers_uncovered = sum(1 for nb in nbs if covered_count[nb] == 0)

            if covers_uncovered > 0:
                delta_energy = 1 - covers_uncovered * 1000
            else:
                delta_energy = 1

            # Accept with probability exp(-delta_energy / T) if delta_energy > 0
            if delta_energy <= 0 or random.random() < np.exp(-delta_energy / T):
                # Apply change
                selected[idx] = True
                for nb in nbs:
                    covered_count[nb] += 1
                new_energy = current_energy + delta_energy
                current_energy = new_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = selected.copy()

        T *= cooling_rate
        step += 1

    # Extract best solution
    best_code_ints = [i for i in range(total_words) if best_selected[i]]

    # Convert to words
    best_code = [int_to_word(i) for i in best_code_ints]

    # Verify full coverage and repair if needed
    # Recompute coverage from best_code
    final_covered = np.zeros(total_words, dtype=np.int32)
    for c_int in best_code_ints:
        for nb in neighbors_table[c_int]:
            final_covered[nb] += 1

    # Add any missing codewords for uncovered words
    uncovered_words = np.where(final_covered == 0)[0]
    for w_int in uncovered_words:
        word = int_to_word(w_int)
        if not best_selected[w_int]:
            best_code.append(word)
            best_selected[w_int] = True
            for nb in neighbors_table[w_int]:
                final_covered[nb] += 1

    # Final greedy removal pass
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        removed = False
        for idx in range(len(best_code) - 1, -1, -1):
            c_int = word_to_int(best_code[idx])
            nbs = neighbors_table[c_int]
            can_remove = all(final_covered[nb] > 1 for nb in nbs)
            if can_remove:
                for nb in nbs:
                    final_covered[nb] -= 1
                best_code.pop(idx)
                best_selected[c_int] = False
                removed = True
                break
        if not removed:
            break

    return best_code
# EVOLVE-BLOCK-END
