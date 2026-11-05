import numpy as np
import time
import itertools
import random
import sys

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
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

            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    # Bit-parallel representation
    num_words = (total_points + 63) // 64  # 103
    masks = np.zeros((total_points, num_words), dtype=np.uint64)
    for idx in range(total_points):
        for nbr in neighbors[idx]:
            word_idx = nbr // 64
            bit_idx = nbr % 64
            masks[idx, word_idx] |= (1 << bit_idx)

    # Full coverage mask
    full_mask = np.full(num_words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    remainder = total_points % 64
    if remainder != 0:
        full_mask[-1] = (1 << remainder) - 1

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def popcount_uint64(x):
        """Popcount for uint64 array using numpy."""
        if hasattr(np, 'bitwise_count'):
            return np.bitwise_count(x)
        # Fallback: use lookup table for 8-bit chunks
        x = x.astype(np.uint64)
        # Split into 8-bit chunks and use lookup
        result = np.zeros(len(x), dtype=np.int32)
        for shift in range(0, 64, 8):
            chunk = (x >> np.uint64(shift)) & np.uint64(0xFF)
            # Lookup table for 8-bit popcount
            lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
            result += lut[chunk]
        return result

    def greedy_set_cover(seed=None, deterministic=False, weighted=False):
        """Bit-parallel greedy: select codewords by maximizing uncovered bits covered."""
        if deterministic:
            rng = None
        else:
            rng = np.random.default_rng(seed)

        covered = np.zeros(num_words, dtype=np.uint64)
        selected = []

        # For weighted greedy, precompute "difficulty" of each point
        # (how many codewords cover it) — points covered by fewer codewords are harder
        if weighted:
            point_coverage = np.zeros(total_points, dtype=np.int32)
            for idx in range(total_points):
                point_coverage[balls[idx]] += 1
            # Inverse: harder points get higher weight
            difficulty = np.zeros(total_points, dtype=np.float64)
            for p in range(total_points):
                if point_coverage[p] > 0:
                    difficulty[p] = 1.0 / point_coverage[p]
                else:
                    difficulty[p] = 1.0
        else:
            difficulty = None

        while not np.array_equal(covered, full_mask):
            # Compute uncovered bits
            uncovered = full_mask ^ (full_mask & covered)

            # For each candidate, count how many uncovered bits it covers
            new_counts = np.zeros(total_points, dtype=np.int32)
            for w in range(num_words):
                if uncovered[w] != 0:
                    masked = masks[:, w] & np.uint64(uncovered[w])
                    new_counts += popcount_uint64(masked).astype(np.int32)

            if weighted and difficulty is not None:
                # Weight by difficulty of newly covered points
                # For each candidate, sum difficulty of newly covered points
                # This is expensive, so we approximate: weight by average difficulty
                # Actually, let's just use the count for now but break ties by difficulty
                pass

            max_count = int(np.max(new_counts))
            if max_count == 0:
                break
            candidates = np.where(new_counts == max_count)[0]
            if len(candidates) > 1:
                if deterministic:
                    best_idx = int(candidates[0])
                else:
                    # Weighted random selection: prefer candidates that cover harder points
                    if weighted and difficulty is not None:
                        # Compute weighted score for each candidate
                        scores = np.zeros(len(candidates), dtype=np.float64)
                        for ci, cand in enumerate(candidates):
                            # Sum difficulty of newly covered points
                            s = 0.0
                            for w in range(num_words):
                                if uncovered[w] != 0:
                                    m = int(masks[cand, w]) & int(uncovered[w])
                                    if m:
                                        # Count bits and weight by difficulty
                                        # This is expensive; approximate
                                        s += bin(m).count('1')
                            scores[ci] = s
                        # Use softmax-like selection
                        scores = scores / (np.max(scores) + 1e-9)
                        probs = scores / np.sum(scores)
                        best_idx = int(candidates[rng.choice(len(candidates), p=probs)])
                    else:
                        best_idx = int(rng.choice(candidates))
            else:
                best_idx = int(candidates[0])
            selected.append(best_idx)
            covered |= masks[best_idx]

        return selected

    def prune(selected_indices):
        """Remove redundant codewords iteratively."""
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

    def rebuild_coverage(selected_list):
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return coverage_count

    def is_valid_code(selected_list):
        if not selected_list:
            return total_points == 0
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1
        return bool(np.all(coverage_count > 0))

    def local_search_walk(selected_indices, deadline, max_swaps=None):
        """Enhanced local search with multiple move types."""
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        # Temperature for simulated annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995

        while time.time() < deadline:
            # 1. Remove any removable codeword (full sweep, deterministic)
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # 2. Try various move types
            move_accepted = False

            # Type A: Single neighbor move (move codeword to a point in its ball)
            for _ in range(20):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)

                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue

                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1

                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue

                    # Valid move
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    move_accepted = True
                    break

                if move_accepted:
                    break

            # Type B: Two-opt move (swap two codewords)
            if not move_accepted and n_sel >= 2:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(n_sel)
                    j = rng.randrange(n_sel)
                    if i == j:
                        continue
                    c_i = selected_list[i]
                    c_j = selected_list[j]

                    # Try moving c_i to a neighbor
                    nbrs_i = list(neighbors[c_i])
                    rng.shuffle(nbrs_i)
                    for c_i_prime in nbrs_i:
                        if c_i_prime == c_i or c_i_prime in selected_set:
                            continue

                        # Tentative: remove c_i, add c_i_prime
                        coverage_count[balls[c_i]] -= 1
                        coverage_count[balls[c_i_prime]] += 1

                        if np.any(coverage_count[balls[c_i]] == 0):
                            coverage_count[balls[c_i_prime]] -= 1
                            coverage_count[balls[c_i]] += 1
                            continue

                        # Now try to also move c_j
                        nbrs_j = list(neighbors[c_j])
                        rng.shuffle(nbrs_j)
                        for c_j_prime in nbrs_j:
                            if c_j_prime == c_j or c_j_prime in selected_set or c_j_prime == c_i_prime:
                                continue

                            coverage_count[balls[c_j]] -= 1
                            coverage_count[balls[c_j_prime]] += 1

                            if np.any(coverage_count[balls[c_j]] == 0):
                                coverage_count[balls[c_j_prime]] -= 1
                                coverage_count[balls[c_j]] += 1
                                continue

                            # Both moves valid
                            selected_list[i] = c_i_prime
                            selected_list[j] = c_j_prime
                            selected_set.discard(c_i)
                            selected_set.discard(c_j)
                            selected_set.add(c_i_prime)
                            selected_set.add(c_j_prime)
                            move_accepted = True
                            break

                        if move_accepted:
                            break

                        # Revert c_j move if not accepted
                        if not move_accepted:
                            coverage_count[balls[c_j_prime]] -= 1
                            coverage_count[balls[c_j]] += 1

                        # Revert c_i move if c_j move not accepted
                        if not move_accepted:
                            coverage_count[balls[c_i_prime]] -= 1
                            coverage_count[balls[c_i]] += 1

                        if move_accepted:
                            break

                    if move_accepted:
                        break

            # Type C: Insert new codeword, then remove two old ones
            if not move_accepted and n_sel >= 2:
                # Pick a random uncovered or under-covered point as candidate
                # Actually, pick a random point not in selected_set
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    c_new = rng.randrange(total_points)
                    if c_new in selected_set:
                        continue

                    # Tentative insert
                    coverage_count[balls[c_new]] += 1

                    # Try to remove two codewords
                    removed_count = 0
                    removed_indices = []
                    for i in range(len(selected_list)):
                        idx = selected_list[i]
                        if np.all(coverage_count[balls[idx]] > 1):
                            coverage_count[balls[idx]] -= 1
                            removed_indices.append(i)
                            removed_count += 1
                            if removed_count == 2:
                                break

                    if removed_count == 2:
                        # Accept: remove the two, add the new
                        # Remove in reverse order to preserve indices
                        for i in reversed(removed_indices):
                            removed_idx = selected_list[i]
                            selected_set.discard(removed_idx)
                            selected_list.pop(i)
                        selected_list.append(c_new)
                        selected_set.add(c_new)
                        move_accepted = True
                    else:
                        # Revert
                        coverage_count[balls[c_new]] -= 1

                    if move_accepted:
                        break

            # After any move, prune
            if move_accepted:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = rebuild_coverage(pruned)
                    selected_list = pruned
                    selected_set = set(selected_list)

                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min

            # If no move accepted, do a random restart-like perturbation
            # Occasionally, add a random codeword to escape local minima
            if not move_accepted and rng.random() < 0.1 and T > T_min:
                c_rand = rng.randrange(total_points)
                if c_rand not in selected_set:
                    coverage_count[balls[c_rand]] += 1
                    selected_list.append(c_rand)
                    selected_set.add(c_rand)
                    # Now try to remove one
                    for i in range(len(selected_list)):
                        idx = selected_list[i]
                        if np.all(coverage_count[balls[idx]] > 1):
                            coverage_count[balls[idx]] -= 1
                            selected_list.pop(i)
                            selected_set.discard(idx)
                            break

        return best_code

    def polish(selected_indices, deadline):
        """Final high-precision polish with exhaustive local search."""
        selected_list = list(selected_indices)
        coverage_count = rebuild_coverage(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]

        # 1. Aggressive deterministic removal
        for _ in range(200):
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
            if not changed:
                break

        # 2. Exhaustive single-move search with removal
        rng = random.Random(99)
        attempts = 0
        max_attempts = 20000

        while time.time() < deadline and attempts < max_attempts:
            if not selected_list:
                break

            # Try each codeword in order (deterministic sweep)
            for i in range(len(selected_list)):
                if time.time() > deadline:
                    break
                c = selected_list[i]
                nbrs = list(neighbors[c])

                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue

                    # Tentative move
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1

                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue

                    # Move is valid. Check if we can remove any codeword.
                    removed_any = False
                    removed_idx = -1
                    for j in range(len(selected_list)):
                        idx_j = selected_list[j]
                        if np.all(coverage_count[balls[idx_j]] > 1):
                            removed_any = True
                            removed_idx = j
                            break

                    if removed_any:
                        val = selected_list[removed_idx]
                        coverage_count[balls[val]] -= 1
                        selected_list.pop(removed_idx)
                        selected_set.discard(val)

                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]

                        attempts += 1
                        break
                    else:
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        attempts += 1
                        break

                if removed_any:
                    break

            if not selected_list:
                break

        # 3. Final prune
        selected_list = prune(selected_list)
        if len(selected_list) < best_len:
            best_len = len(selected_list)
            best_code = selected_list[:]

        return best_code

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate initial codes
    best_code = None
    best_len = float('inf')

    # Deterministic greedy
    selected = greedy_set_cover(deterministic=True)
    selected = prune(selected)
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected[:]

    # Weighted greedy
    selected = greedy_set_cover(deterministic=True, weighted=True)
    selected = prune(selected)
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected[:]

    # Random greedy with multiple seeds
    for seed in range(8):
        if time.time() > deadline - 50:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

        # Also try weighted
        selected = greedy_set_cover(seed, weighted=True)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_set_cover(deterministic=True)
        best_code = prune(best_code)
        best_len = len(best_code)

    # 2. Local Search on best code
    ls_deadline = deadline - 20
    best_code = local_search_walk(best_code, ls_deadline)

    # 3. Polish
    best_code = polish(best_code, deadline)

    # Verify validity
    if not is_valid_code(best_code):
        best_code = greedy_set_cover(deterministic=True)
        best_code = prune(best_code)

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
