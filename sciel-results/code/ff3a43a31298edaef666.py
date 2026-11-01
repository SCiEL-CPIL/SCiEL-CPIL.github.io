"""Optimized covering code search for K_3(6,1) using bitmask neighbors and targeted swaps."""
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

    # Precompute index <-> word conversions
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (distance <= radius) for every word in the space
    # Each word has 1 + n*(q-1) = 1 + 6*2 = 13 neighbors
    neighbors = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = [i]  # include self
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.append(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)

    # Build bitmask representation: each word's neighbor set as a Python int bitmask
    # 729 bits fits in a Python int
    neighbor_masks = []
    for i in range(space_size):
        mask = 0
        for nb in neighbors[i]:
            mask |= (1 << nb)
        neighbor_masks.append(mask)

    # Precompute overlap counts: how many neighbors two codewords share
    # overlap[i][j] = |neighbors[i] ∩ neighbors[j]|
    # This is a 729x729 matrix, but we can compute on the fly with bitmasks
    # Actually, let's precompute it as a numpy array for speed
    overlap = np.zeros((space_size, space_size), dtype=np.int8)
    for i in range(space_size):
        mi = neighbor_masks[i]
        for j in range(i + 1, space_size):
            mj = neighbor_masks[j]
            ov = bin(mi & mj).count('1')
            overlap[i, j] = ov
            overlap[j, i] = ov

    # Tetracode direct-sum initial construction: 9 * 3^(6-4) = 81 words
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]

    def build_counts(codeword_indices):
        """Build coverage count array from a list of codeword indices."""
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in codeword_indices:
            counts[neighbors[ci]] += 1
        return counts

    def is_redundant(ci, counts):
        """Check if codeword ci can be removed (no neighbor has count == 1)."""
        for nb in neighbors[ci]:
            if counts[nb] == 1:
                return False
        return True

    def try_remove(current, counts, deadline):
        """Try to remove redundant codewords. Returns (current, counts, removed_any)."""
        removed = False
        i = 0
        while i < len(current):
            if time.time() > deadline:
                break
            ci = current[i]
            if is_redundant(ci, counts):
                for nb in neighbors[ci]:
                    counts[nb] -= 1
                current.pop(i)
                removed = True
            else:
                i += 1
        return current, counts, removed

    def local_search(cw_indices, counts, deadline, max_iters=50000):
        """
        Local search: try swaps to enable further deletions.
        Strategy: For each codeword, find its essential neighbors (count==1).
        Try replacing it with a candidate that covers all essential neighbors
        and doesn't leave any word uncovered.
        """
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        swaps_made = 0

        while swaps_made < max_iters and time.time() < deadline:
            progress = False

            # Phase 1: try simple removals
            current, cnt, removed = try_remove(current, cnt, deadline)
            if removed:
                continue

            # Phase 2: try swaps
            # For each codeword, find essential neighbors
            best_swap = None
            best_score = -1

            for i in range(len(current)):
                if time.time() > deadline:
                    break
                ci = current[i]
                essential = [nb for nb in neighbors[ci] if cnt[nb] == 1]
                if not essential:
                    # Should have been caught in phase 1
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
                    break

                # Try a few random candidates
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    cand = random.randint(0, space_size - 1)
                    if cand in current_set:
                        continue
                    cand_mask = neighbor_masks[cand]
                    # Check candidate covers all essential neighbors
                    covers = True
                    for e in essential:
                        if not (cand_mask >> e) & 1:
                            covers = False
                            break
                    if not covers:
                        continue
                    # Check: after swap, no word becomes uncovered
                    # Words losing coverage: in neighbors[ci] but not in cand_nbs, with cnt==1
                    valid = True
                    ci_mask = neighbor_masks[ci]
                    losing = ci_mask & ~cand_mask
                    # Check if any losing word has cnt==1
                    losing_bits = losing
                    while losing_bits:
                        lsb = losing_bits & -losing_bits
                        pos = lsb.bit_length() - 1
                        if cnt[pos] == 1:
                            valid = False
                            break
                        losing_bits ^= lsb
                    if not valid:
                        continue
                    # Score this swap: how much extra coverage does it provide?
                    gaining = cand_mask & ~ci_mask
                    gain = bin(gaining).count('1')
                    if gain > best_score:
                        best_score = gain
                        best_swap = (i, ci, cand)

            if best_swap is not None:
                i, ci, cand = best_swap
                for nb in neighbors[ci]:
                    cnt[nb] -= 1
                for nb in neighbors[cand]:
                    cnt[nb] += 1
                current[i] = cand
                current_set.discard(ci)
                current_set.add(cand)
                swaps_made += 1
                progress = True

            if not progress:
                break

        return current, cnt

    def try_randomized_removal_order(cw_indices, counts, deadline, seed):
        """Try removal in a randomized order, then swap local search."""
        rng = random.Random(seed)
        current = list(cw_indices)
        cnt = counts.copy()

        # Randomized removal passes
        for _ in range(20):
            if time.time() > deadline:
                break
            rng.shuffle(current)
            removed = False
            for ci in current[:]:
                if time.time() > deadline:
                    break
                if is_redundant(ci, cnt):
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.remove(ci)
                    removed = True
            if not removed:
                break

        # Then swap local search
        current, cnt = local_search(current, cnt, deadline, max_swaps=8000)
        return current, cnt

    # Main search loop with multiple restarts
    best_size = float('inf')
    best_code = None

    # Initial: deterministic greedy remove + swap
    cw_indices = [word_to_idx(w) for w in initial_codewords]
    counts = build_counts(cw_indices)
    code, counts, _ = try_remove(cw_indices, counts, deadline - 30)
    code, counts = local_search(code, counts, deadline - 15, max_swaps=15000)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # Multiple randomized restarts
    remaining_deadline = deadline - 5
    restarts = 0
    while time.time() < remaining_deadline:
        restarts += 1
        seed = restarts * 7919 + 13
        code_r, counts_r = try_randomized_removal_order(cw_indices, counts, remaining_deadline, seed)
        if len(code_r) < best_size:
            best_size = len(code_r)
            best_code = code_r.copy()
        # If we've hit the known optimum, stop
        if best_size <= 73:
            break

    # Final polish: one more greedy remove + swap on the best
    if best_code is not None and time.time() < deadline - 3:
        final_counts = build_counts(best_code)
        final_code, final_counts, _ = try_remove(best_code, final_counts, deadline - 2)
        final_code, final_counts = local_search(final_code, final_counts, deadline, max_swaps=10000)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)

    # Convert back to list of lists
    result = [idx_to_word(ci) for ci in best_code]
    return result


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Precompute coverage sets
    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
