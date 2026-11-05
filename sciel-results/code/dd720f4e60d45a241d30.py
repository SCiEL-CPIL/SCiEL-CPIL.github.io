"""Combined program: vectorized greedy + enhanced local search + simulated annealing polish for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Total number of words in the space
    total_words = q ** n

    # Build index mapping: integer -> tuple of digits, and tuple -> integer
    word_to_idx = {}
    idx_to_word = {}
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= q
        word_to_idx[tuple(digits)] = i
        idx_to_word[i] = tuple(digits)

    # Precompute neighbors (ball of radius 1) for each word as list of indices
    neighbors = []
    neighbor_sets = []
    for i in range(total_words):
        w = list(idx_to_word[i])
        neigh = [i]
        if radius == 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    w[pos] = val
                    idx = word_to_idx[tuple(w)]
                    neigh.append(idx)
                w[pos] = orig_val
        else:
            def _gen(idx, current, changes_left):
                if changes_left == 0:
                    neigh.append(word_to_idx[tuple(current)])
                    return
                _gen(idx + 1, current, changes_left)
                orig = current[idx]
                for val in range(q):
                    if val != orig:
                        current[idx] = val
                        _gen(idx + 1, current, changes_left - 1)
                current[idx] = orig
            _gen(0, list(idx_to_word[i]), radius)
            neigh = list(set(neigh))
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))

    # Convert neighbors to numpy arrays for vectorized operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    # Phase 1: Greedy Set Cover with vectorized gain computation
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 40:
            break

        gains = np.zeros(total_words, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbor_arrays[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False

    # Phase 2: Local Search - deletion pass and swap pass
    counts = np.zeros(total_words, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    # Iterative local search
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False

        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if np.min(counts[neighbor_arrays[c]]) > 1:
                to_remove.append(c)

        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # --- Swap Pass ---
        if time.time() > deadline - 20:
            break

        random.shuffle(code)

        for c in code:
            if time.time() > deadline - 20:
                break

            if np.min(counts[neighbor_arrays[c]]) > 1:
                continue

            crit_points = [i for i in neighbors[c] if counts[i] == 1]

            if not crit_points:
                continue

            found_swap = False
            random.shuffle(crit_points)

            for p in crit_points:
                if found_swap:
                    break

                cands = [x for x in neighbors[p] if x not in code_set]

                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break

                    if not valid:
                        continue

                    score = int(np.sum(counts[neighbor_arrays[w]] == 2))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbor_arrays[best_w]] += 1

                    improved = True
                    found_swap = True
                    break

            if not found_swap:
                temp_counts = counts.copy()
                temp_counts[neighbor_arrays[c]] -= 1

                if np.min(temp_counts) >= 1:
                    continue

                would_be_uncovered = [i for i in neighbors[c] if temp_counts[i] == 0]

                if not would_be_uncovered:
                    continue

                all_needed = set(crit_points) | set(would_be_uncovered)

                best_w = -1
                best_score = -1

                first_point = list(all_needed)[0]
                cands = [x for x in neighbors[first_point] if x not in code_set]

                for w in cands:
                    valid = True
                    for k in all_needed:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue

                    score = int(np.sum(counts[neighbor_arrays[w]] == 2))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbor_arrays[best_w]] += 1

                    improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

    # Phase 3: Simulated Annealing polish on the best solution found
    if time.time() < deadline - 15:
        # Start from best_code
        sa_code = list(best_code)
        sa_code_set = set(sa_code)
        sa_counts = np.zeros(total_words, dtype=np.int32)
        for idx in sa_code:
            sa_counts[neighbor_arrays[idx]] += 1

        sa_size = len(sa_code)
        sa_best_size = sa_size
        sa_best_code = list(sa_code)

        # SA parameters
        T_initial = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        max_sa_time = deadline - 12 - time.time()

        if max_sa_time > 2:
            T = T_initial
            sa_improved = True
            while sa_improved and T > T_min and time.time() < deadline - 12:
                sa_improved = False
                
                # Deletion pass first
                to_remove = []
                for c in sa_code:
                    if np.min(sa_counts[neighbor_arrays[c]]) > 1:
                        to_remove.append(c)
                
                if to_remove:
                    sa_improved = True
                    for c in to_remove:
                        sa_code.remove(c)
                        sa_code_set.remove(c)
                        sa_counts[neighbor_arrays[c]] -= 1
                
                if len(sa_code) < sa_best_size:
                    sa_best_size = len(sa_code)
                    sa_best_code = list(sa_code)
                    best_size = sa_best_size
                    best_code = list(sa_best_code)

                # SA move: try to add a random word and then remove a redundant one
                if time.time() > deadline - 12:
                    break

                # Randomly pick a candidate word to add
                cand = random.randrange(total_words)
                if cand in sa_code_set:
                    continue

                # Add candidate
                sa_code.append(cand)
                sa_code_set.add(cand)
                sa_counts[neighbor_arrays[cand]] += 1

                # Now try to remove a word that became redundant
                removed = False
                # Shuffle to randomize removal order
                removal_order = list(sa_code)
                random.shuffle(removal_order)
                
                for c in removal_order:
                    if np.min(sa_counts[neighbor_arrays[c]]) > 1:
                        sa_code.remove(c)
                        sa_code_set.remove(c)
                        sa_counts[neighbor_arrays[c]] -= 1
                        removed = True
                        break

                # If we removed a word, we've made a move
                if removed:
                    new_size = len(sa_code)
                    # We added one and removed one, so size is same as before
                    # Accept if size decreased or with probability based on T
                    # Actually size is unchanged (added 1, removed 1)
                    # The benefit is potentially better structure for future moves
                    # Let's just accept and continue
                    sa_improved = True
                else:
                    # Could not remove, so revert the addition
                    sa_code.remove(cand)
                    sa_code_set.remove(cand)
                    sa_counts[neighbor_arrays[cand]] -= 1

                # Update best
                if len(sa_code) < sa_best_size:
                    sa_best_size = len(sa_code)
                    sa_best_code = list(sa_code)
                    best_size = sa_best_size
                    best_code = list(sa_best_code)

                T *= cooling_rate

    # Phase 4: Final deletion pass on best_code
    final_counts = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        final_counts[neighbor_arrays[idx]] += 1
    
    final_code = list(best_code)
    final_code_set = set(final_code)
    
    # Multiple deletion passes until no more removals
    for _ in range(5):
        changed = False
        for c in list(final_code):
            if np.min(final_counts[neighbor_arrays[c]]) > 1:
                final_code.remove(c)
                final_code_set.remove(c)
                final_counts[neighbor_arrays[c]] -= 1
                changed = True
        if not changed:
            break
    
    if len(final_code) < best_size:
        best_size = len(final_code)
        best_code = final_code

    # Convert best_code back to list of lists
    result = [list(idx_to_word[cw]) for cw in best_code]

    return result
# EVOLVE-BLOCK-END
