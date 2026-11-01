"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 140  # 140s safety margin out of 150s

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # For the specific task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        # Use vectorized greedy algorithm
        best_code = _greedy_covering_code(3, 7, 1, deadline)
        if best_code is not None:
            return best_code
    return initial


def _greedy_covering_code(q, n, radius, deadline):
    """Vectorized greedy covering code construction."""
    # Total space size
    total = q ** n  # 2187 for q=3, n=7

    # Represent all words as integers 0..total-1
    all_words = np.arange(total, dtype=np.int32)

    # Precompute Hamming distance from each word to each other word
    # This is a total x total matrix, which is 2187 x 2187 ~ 4.8M entries
    # We can compute distances efficiently using base-q digit comparison

    # Convert all words to digit arrays
    digits = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        digits[:, i] = (all_words // (q ** (n - 1 - i))) % q

    # Compute pairwise Hamming distances
    # dist[i, j] = sum over positions where digits[i, :] != digits[j, :]
    # Using broadcasting: for each position, count mismatches
    dist = np.zeros((total, total), dtype=np.int8)
    for i in range(n):
        # For position i, digits[:, i] is shape (total,)
        # Mismatch matrix: digits[:, i][:, None] != digits[:, i][None, :]
        dist += (digits[:, i][:, None] != digits[:, i][None, :]).astype(np.int8)

    # Greedy: start with empty code, repeatedly add word that covers most uncovered
    covered = np.zeros(total, dtype=bool)
    code = []

    # Precompute for each word, the list of words it covers (distance <= radius)
    # covers[i] = set of j where dist[i, j] <= radius
    covers_mask = dist <= radius  # shape (total, total)

    # Track uncovered count for each candidate
    uncovered_count = np.sum(~covered, axis=0) if False else None

    # Simpler approach: maintain a count of how many uncovered words each candidate covers
    # Initialize: for each candidate i, count of uncovered words in its cover
    # covered is all False initially, so count[i] = number of j where covers_mask[i,j] is True
    cover_counts = np.sum(covers_mask, axis=1).astype(np.int32)

    while True:
        # Check deadline
        if time.time() > deadline - 5:
            break

        if not np.any(~covered):
            break

        # Find candidate with max cover count
        best_idx = int(np.argmax(cover_counts))
        if cover_counts[best_idx] == 0:
            break

        code.append(list(digits[best_idx]))
        covered |= covers_mask[best_idx]

        # Update cover counts: for each other candidate, subtract the newly covered words
        # that it also covers
        # new_covered = covers_mask[best_idx] & ~old_covered
        # For each candidate i, cover_counts[i] -= sum over j in new_covered where covers_mask[i,j]
        # This is: cover_counts -= covers_mask[:, best_idx] & (covers_mask[best_idx] & ~covered_before)
        # But we already updated covered, so we need to track what was newly covered

        # More efficient: recompute is too slow. Instead:
        # newly_covered = covers_mask[best_idx] & ~covered  (before updating covered)
        # But we already did covered |= covers_mask[best_idx]
        # So we need to do this before the update, or compute differently

        # Let's restructure: compute newly_covered before updating
        # Actually, let's just recompute cover_counts periodically or use a different approach

        # For now, recompute cover counts when code size is small, or use incremental
        # Incremental: for each i, cover_counts[i] = number of j where covers_mask[i,j] and not covered[j]
        # = original_count[i] - number of j in covered where covers_mask[i,j]
        # This requires tracking covered per candidate, which is expensive

        # Alternative: just recompute from scratch every step (slow but simple)
        # Or: since total is only 2187, and we expect ~186 codewords, that's 186 * 2187 * 7 ~ 2.8M ops
        # Let's just recompute each time for simplicity, but check deadline

        if time.time() > deadline - 10:
            break

        # Recompute cover counts
        ~covered  # This is the current uncovered mask
        # cover_counts[i] = sum over j of (covers_mask[i,j] and not covered[j])
        # = sum over j of covers_mask[i,j] * (~covered[j])
        cover_counts = (covers_mask * (~covered).astype(np.int8)).sum(axis=1).astype(np.int32)

    return code if code else None
# EVOLVE-BLOCK-END
