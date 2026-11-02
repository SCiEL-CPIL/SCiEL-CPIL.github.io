"""Covering code K_3(7,1) via greedy set cover with bitset acceleration and time guard."""
# EVOLVE-BLOCK-START
import time
import itertools

# Deadline: evaluator kills at 150s; we return well before that.
_DEADLINE = 140.0  # seconds


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # For the specific task: q=3, n=7, radius=1
    # Total space: 3^7 = 2187 words
    # Ball size (radius 1): 1 + 7*(3-1) = 1 + 14 = 15 words per ball

    # Precompute all words as integers 0..q^n - 1
    total = q ** n
    # Map integer -> tuple for output
    # We'll work with integers for speed

    # Precompute the ball (set of words within distance <= radius) for each possible center
    # Ball size: sum_{i=0}^{radius} C(n,i) * (q-1)^i
    # For q=3, n=7, radius=1: 1 + 7*2 = 15

    # Represent each word as a tuple of digits
    # Precompute digit arrays
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))

    # Precompute powers for index calculation
    powers = [q ** (n - 1 - i) for i in range(n)]

    # For each center, compute ball mask
    # ball_masks[center_idx] is a bitset of all words within Hamming distance <= radius
    ball_masks = [0] * total
    for center_idx in range(total):
        center = all_words[center_idx]
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                # For each position, we can change the digit to any other value
                # Instead of iterating all q digits and skipping the current one,
                # we can just iterate all q digits, but skipping is cheap.
                for alt in range(q):
                    if alt == center[pos]:
                        continue
                    new_idx = center_idx + (alt - center[pos]) * powers[pos]
                    mask |= (1 << new_idx)
        # If radius > 1, we would need to add pairs of changes, etc.
        # For this task (radius 1), this is sufficient.
        ball_masks[center_idx] = mask

    # Greedy set cover
    # uncovered: bitset of all words
    all_mask = (1 << total) - 1
    uncovered = all_mask

    # For each candidate center, precompute its ball mask (done above)
    # Greedy: repeatedly pick the center that covers the most uncovered words

    # To speed up, we can precompute ball_masks and use bit operations
    # But 2187 candidates * 2187 bits = manageable

    # We'll use a list of (covered_count, center_idx) and update
    # Actually, for 2187 candidates, we can just recompute the count each iteration
    # Number of iterations: at most 2187/15 ≈ 146 (lower bound), best known 186
    # Each iteration: 2187 bitset ANDs of 2187 bits = 2187 * (2187/64) word ops ≈ 74k ops
    # 186 iterations * 74k = ~14M ops, very fast in Python with big ints

    selected = []
    remaining_candidates = list(range(total))

    while uncovered != 0:
        if time.time() > deadline - 5:  # Safety margin
            break

        best_count = -1
        best_idx = -1
        # Iterate over remaining candidates
        # We can optimize by breaking early if we find a candidate that covers all remaining?
        # Unlikely to happen, but cnt == 0 break is not useful for finding max.
        for c in remaining_candidates:
            # Count uncovered words covered by center c
            covered = ball_masks[c] & uncovered
            # Use bit_count() for speed (Python 3.10+)
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                # If we cover everything, we are done
                if cnt == uncovered.bit_count():
                    break

        if best_idx == -1 or best_count == 0:
            # Shouldn't happen if we have a valid covering
            break

        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        # Remove best_idx from remaining_candidates
        # List removal is O(N), but N is small (2187). 
        # To be faster, we could use a set, but iteration order doesn't matter for greedy max.
        # However, list removal is fine.
        try:
            remaining_candidates.remove(best_idx)
        except ValueError:
            pass

    # Check if we have a valid covering
    if uncovered != 0:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert selected indices back to lists of digits
    result = []
    for idx in selected:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)

    return result
# EVOLVE-BLOCK-END
