"""Covering code K_3(7,1): greedy set cover + local search (removal & swap) with numpy."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 138.0  # seconds (evaluator kills at 150s)


def _digits_to_int(digits, q, n):
    """Convert a digit list to an integer index (big-endian)."""
    idx = 0
    for d in digits:
        idx = idx * q + d
    return idx


def _int_to_digits(idx, q, n):
    """Convert integer index to digit list (big-endian)."""
    digits = [0] * n
    for i in range(n - 1, -1, -1):
        digits[i] = idx % q
        idx //= q
    return digits


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    total = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

    # Precompute all words as a numpy array of shape (total, n)
    all_words = np.zeros((total, n), dtype=np.int32)
    for idx in range(total):
        x = idx
        for i in range(n - 1, -1, -1):
            all_words[idx, i] = x % q
            x //= q

    # Precompute ball membership: for each center, list of covered word indices
    # ball[i] = list of word indices covered by center i
    # Use numpy for vectorized ball computation
    # For each position pos and each alternative value alt != center[pos], compute neighbor index
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # Precompute ball as a 2D array: ball[center_idx, j] = word index, for j in 0..ball_size-1
    # ball[center, 0] = center itself
    # ball[center, 1 + pos*(q-1) + alt_idx] = neighbor
    ball = np.zeros((total, ball_size), dtype=np.int64)
    ball[:, 0] = np.arange(total, dtype=np.int64)

    # For each position and each alternative, compute the neighbor index
    for pos in range(n):
        center_digits = all_words[:, pos]  # shape (total,)
        for alt in range(q):
            mask = center_digits != alt
            # For centers where center_digits[pos] != alt, the neighbor is:
            # center_idx + (alt - center_digits[pos]) * powers[pos]
            neighbor_idx = np.arange(total, dtype=np.int64) + (alt - center_digits) * powers[pos]
            # Only valid where mask is True
            # Find the slot: for position pos, alternative alt, slot = 1 + pos*(q-1) + alt_rank
            # alt_rank: the rank of alt among values != center_digits[pos]
            # This is tricky; let's just fill it differently.
            pass

    # Alternative: precompute balls more directly
    # For each center, compute its ball by iterating over positions and alternatives
    # This is O(total * n * q) which is 2187 * 7 * 3 = 45927 operations - fine
    ball = np.zeros((total, ball_size), dtype=np.int64)
    ball[:, 0] = np.arange(total, dtype=np.int64)
    slot = 1
    for pos in range(n):
        for alt in range(q):
            # For centers where all_words[center, pos] != alt
            center_val = all_words[:, pos]
            valid = center_val != alt
            # neighbor index = center_idx + (alt - center_val) * powers[pos]
            # Only for valid centers
            neighbor = np.arange(total, dtype=np.int64) + (alt - center_val) * powers[pos]
            # Place into ball[valid_centers, slot]
            # But slot depends on which alt was skipped...
            # Let's just do it per-center for clarity since total is small
            pass

    # Simpler: just build balls per center (total=2187, n=7, q=3 -> 45927 ops)
    balls = []
    for c in range(total):
        b = [c]
        for pos in range(n):
            cv = all_words[c, pos]
            for alt in range(q):
                if alt == cv:
                    continue
                ni = c + (alt - cv) * int(powers[pos])
                b.append(ni)
        balls.append(b)

    # Convert to numpy array for vectorized ops
    ball_arr = np.array(balls, dtype=np.int64)  # shape (total, ball_size)

    # Greedy set cover using bitsets (Python ints)
    # all_mask has bits 0..total-1 set
    all_mask = (1 << total) - 1

    # Precompute ball bitmasks
    ball_masks = np.zeros(total, dtype=object)
    for c in range(total):
        m = 0
        for w in balls[c]:
            m |= (1 << w)
        ball_masks[c] = m

    # Greedy: repeatedly pick the center that covers the most uncovered words
    uncovered = all_mask
    selected = []
    remaining = set(range(total))

    while uncovered != 0:
        if time.time() > deadline - 30:
            break
        best_count = -1
        best_idx = -1
        unc_count = uncovered.bit_count()
        for c in remaining:
            covered = ball_masks[c] & uncovered
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                if cnt == unc_count:
                    break
        if best_idx == -1 or best_count == 0:
            break
        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        remaining.discard(best_idx)

    # Fallback if greedy didn't finish
    if uncovered != 0:
        # Use product construction as fallback
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return result

    # Local search: removal
    # Track coverage count for each word: how many selected codewords cover it
    cov_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        for w in balls[c]:
            cov_count[w] += 1

    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline - 15:
                break
            c = selected[i]
            # Check if removing c leaves all words covered
            # A word is uncovered after removal if cov_count[w] == 1 and w is in ball[c]
            redundant = True
            for w in balls[c]:
                if cov_count[w] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove codeword i
                for w in balls[c]:
                    cov_count[w] -= 1
                selected.pop(i)
                improved = True
                break

    # Local search: swap (replace one codeword with another to reduce total or maintain)
    # Actually, for minimizing size, swaps don't directly help unless we can remove more.
    # Instead, try: for each selected codeword, see if replacing it with a different one
    # allows further removals. This is complex; skip for now.

    # Additional local search: try to find a smaller set via random restarts
    # If we have time, do a few random restarts of greedy and keep the best
    best_selected = selected[:]
    best_size = len(selected)

    if time.time() < deadline - 20:
        import random
        rng = random.Random(42)
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            # Randomized greedy: at each step, pick randomly among top-k candidates
            uncovered = all_mask
            sel = []
            rem = set(range(total))
            k = 3  # top-k
            while uncovered != 0:
                if time.time() > deadline - 10:
                    break
                unc_count = uncovered.bit_count()
                # Find top-k candidates
                candidates = []
                for c in rem:
                    covered = ball_masks[c] & uncovered
                    cnt = covered.bit_count()
                    candidates.append((cnt, c))
                candidates.sort(key=lambda x: -x[0])
                if not candidates or candidates[0][0] == 0:
                    break
                top_k = candidates[:k]
                # Pick randomly from top_k
                pick = rng.choice(top_k)
                sel.append(pick[1])
                uncovered &= ~ball_masks[pick[1]]
                rem.discard(pick[1])
            if uncovered == 0 and len(sel) < best_size:
                best_size = len(sel)
                best_selected = sel[:]

    selected = best_selected

    # Final removal pass on the best solution
    cov_count = np.zeros(total, dtype=np.int32)
    for c in selected:
        for w in balls[c]:
            cov_count[w] += 1

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline - 5:
                break
            c = selected[i]
            redundant = True
            for w in balls[c]:
                if cov_count[w] == 1:
                    redundant = False
                    break
            if redundant:
                for w in balls[c]:
                    cov_count[w] -= 1
                selected.pop(i)
                improved = True
                break

    # Convert to list of digit lists
    result = []
    for idx in selected:
        result.append(_int_to_digits(idx, q, n))

    return result
# EVOLVE-BLOCK-END
