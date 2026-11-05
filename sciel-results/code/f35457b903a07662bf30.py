"""Football pool / ternary covering code K_3(8,1).

Strategy: bit-parallel greedy set-cover using 64-bit masks, then a fast
redundancy-removal (reverse greedy) pass, with a hard deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    # Deadline guard: evaluator kills at 150 s; return best result before ~140 s.
    deadline = time.time() + 130.0

    # For the standard task: q=3, n=8, radius=1.
    # Total space size: 3**8 = 6561.
    total = q ** n

    # Build all words as tuples and index them.
    # Represent each word as a flat index 0..total-1 in base-q representation.
    # Precompute the Hamming-ball of radius `radius` for each word.
    # For radius=1, n=8, q=3: each ball has 1 + 8*2 = 17 points.

    # Build all words as numpy array of shape (total, n) with values in 0..q-1.
    all_words = np.array([
        [int(digit) for digit in str(i).zfill(n)] if False else
        list(_digits(i, q, n))
        for i in range(total)
    ])  # shape (total, n), dtype int64

    # Precompute for each word its ball mask as a set of indices.
    # For radius=1: the ball of word w contains w itself and all words differing
    # in exactly one coordinate.
    # We'll build ball_masks as a list of Python ints (bitmasks over 6561 bits).
    # 6561 bits = 103 64-bit words. We'll use Python's arbitrary-precision ints.

    ball_masks = []
    for i in range(total):
        w = all_words[i]
        mask = (1 << i)  # self
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                # Find the index of the word that differs from w at position pos with value new_val.
                # We need a fast mapping from word to index.
                # Build a dictionary.
                # Actually, let's build the dictionary first.
                pass
        ball_masks.append(mask)

    # Build word-to-index mapping.
    word_to_idx = {}
    for i in range(total):
        word_to_idx[tuple(all_words[i])] = i

    # Rebuild ball_masks properly.
    ball_masks = [0] * total
    for i in range(total):
        w = all_words[i]
        mask = (1 << i)
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                # Modify coordinate pos to new_val.
                w2 = list(w)
                w2[pos] = new_val
                idx2 = word_to_idx[tuple(w2)]
                mask |= (1 << idx2)
        ball_masks[i] = mask

    # Greedy set-cover: repeatedly pick the word whose ball covers the most uncovered points.
    # uncovered = (1 << total) - 1
    full_mask = (1 << total) - 1
    uncovered = full_mask

    selected = []
    # For speed, we'll track uncovered as a Python int and use bitwise operations.
    # Greedy: at each step, find the word maximizing popcount(ball_masks[i] & uncovered).
    # With 6561 candidates and up to ~500 iterations, this is ~3.3M popcount ops on big ints.
    # Python int popcount via bit_count() is fast (C-level).

    # Precompute ball_masks as a numpy array? No, they're big ints. Keep as list.

    # To speed up the greedy, we can batch: use numpy to compute overlaps for all candidates
    # at once? Big ints don't vectorize easily. Alternative: use a set of uncovered indices
    # and count overlaps via set intersection. But 6561 * 500 = 3.3M set intersections of size ~17
    # is also feasible. Let's try the bit_count approach first; if too slow, switch.

    # Actually, for 6561 candidates per iteration and ~500 iterations, that's 3.3M bit_count
    # calls on 6561-bit ints. Each bit_count on a 6561-bit int in Python is O(103) word ops in C.
    # 3.3M * 103 ≈ 340M word ops. In C this is ~0.3-1s. In Python loop overhead, maybe 5-10s.
    # Should fit within budget. Let's proceed.

    start = time.time()
    while uncovered:
        # Check deadline.
        if time.time() > deadline - 5.0:
            break
        # Find best candidate.
        best_idx = -1
        best_count = -1
        for i in range(total):
            if ball_masks[i] & uncovered == 0:
                continue
            c = (ball_masks[i] & uncovered).bit_count()
            if c > best_count:
                best_count = c
                best_idx = i
                if best_count == uncovered.bit_count():
                    break  # perfect: covers all remaining
        if best_idx < 0:
            break
        selected.append(best_idx)
        uncovered &= ~ball_masks[best_idx]

    # Convert selected indices to codewords.
    result = [all_words[i].tolist() for i in selected]

    # Redundancy removal (reverse greedy): try removing each codeword and check coverage.
    # For each selected word in reverse order, check if all points it uniquely covers
    # are still covered by the remaining words.
    # A point is uniquely covered by word i if it's in ball(i) and not in ball(j) for all other j.
    # Equivalently, the union of all other balls still covers the full space.
    # We can check: does (OR of all other ball_masks) == full_mask?

    # Build a list of masks for selected words.
    sel_masks = [ball_masks[i] for i in selected]

    # Compute OR of all masks.
    if sel_masks:
        union_all = 0
        for m in sel_masks:
            union_all |= m

    # Try removing each (in reverse order to prefer keeping later ones? or forward?)
    # Remove in reverse order of selection.
    for k in range(len(sel_masks) - 1, -1, -1):
        if time.time() > deadline - 3.0:
            break
        # Check if removing sel_masks[k] still covers everything.
        # union_without_k = union_all XOR (part only covered by k) — not straightforward.
        # Instead: compute OR of all masks except k.
        # This is O(K) per check, K up to ~500. Total O(K^2) = 250K ORs on 6561-bit ints.
        # Each OR is ~103 word ops in C. 250K * 103 ≈ 26M word ops. Fast.
        u = 0
        for j in range(len(sel_masks)):
            if j == k:
                continue
            u |= sel_masks[j]
        if u == full_mask:
            sel_masks.pop(k)
            selected.pop(k)
            # Recompute union_all for subsequent checks.
            union_all = 0
            for m in sel_masks:
                union_all |= m

    # Final check: verify coverage.
    final_union = 0
    for m in sel_masks:
        final_union |= m
    # For safety, if something went wrong, fall back to the direct-sum construction.
    if final_union != full_mask and selected:
        # Fallback: direct sum of tetracode.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    elif final_union != full_mask:
        # Empty selection means no coverage; use fallback.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert final selected to list of lists.
    result = [all_words[i].tolist() for i in selected]

    return result


def _digits(num, q, n):
    """Return the base-q digits of num as a list of length n (most significant first)."""
    digits = [0] * n
    for pos in range(n - 1, -1, -1):
        digits[pos] = num % q
        num //= q
    return digits
# EVOLVE-BLOCK-END
