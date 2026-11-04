import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    # neighbors[idx] is a list of 17 indices
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def greedy_set_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices

        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break

        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        # Precompute for each point, which codewords cover it (for quick redundancy checks)
        # We'll use a different approach: for each codeword, track which points it uniquely covers

        while time.time() < deadline:
            # Try to remove codewords
            removed = True
            while removed:
                removed = False
                for i in range(len(selected_list)):
                    if time.time() > deadline:
                        break
                    idx = selected_list[i]
                    # Check if any point in this ball has coverage count == 1
                    # If all points have count > 1, we can remove
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        removed = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            # Try swaps: move a codeword to a neighbor position
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # Pick a random codeword and try to move it
            progress = False
            for attempt in range(20):
                if time.time() > deadline:
                    break

                i = rng.randrange(n_sel)
                c = selected_list[i]

                # Try random neighbors of c
                nbrs = neighbors[c]
                rng.shuffle(nbrs)

                for c_prime in nbrs:
                    if c_prime == c:
                        continue
                    # Check if c_prime is already in selected_list
                    if c_prime in selected_list:
                        continue

                    # Tentatively swap
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1

                    # Check validity: all points in balls[c] that are not in balls[c_prime]
                    # must still be covered by other codewords
                    # And all points in balls[c_prime] must be covered

                    # Points that lose coverage: those in balls[c] but not in balls[c_prime]
                    # Points that gain coverage: those in balls[c_prime] but not in balls[c]

                    # Check if any point in balls[c] has count 0 after removal
                    # and any point in balls[c_prime] has count 0 (shouldn't happen since we added)

                    # Efficient check:
                    # Points in balls[c] with count 0 after decrement
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue

                    # Valid swap
                    selected_list[i] = c_prime
                    progress = True
                    break

                if progress:
                    break

            if progress:
                # After a swap, try to prune
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    # Rebuild coverage count
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in pruned:
                        coverage_count[balls[idx]] += 1
                    selected_list = pruned

                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    # Run greedy set cover with different seeds
    best_code = None
    best_len = float('inf')

    for seed in range(8):
        if time.time() > deadline - 50:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local search on the best code
    best_code = local_search_optimize(best_code, deadline)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
