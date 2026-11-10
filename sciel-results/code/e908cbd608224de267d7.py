import time
import itertools
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    # --- Build initial code: direct sum of tetracode with all tails ---
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            code = [list(h) for h in tetracode]
        else:
            code = [list(h) + list(tail) for h in tetracode
                    for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: all-zero word plus greedy
        code = [[0] * n]

    # --- Precompute ball membership ---
    # For each codeword, store set of words it covers (as flat indices)
    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Build candidate pool: all words within distance 1 of any current codeword
    # Actually, for replacement we need candidates that can cover uncovered words.
    # Let's think differently: use a greedy set cover on the full space.

    # Strategy: Greedy set cover
    # 1. Compute all balls (center -> set of covered word indices)
    # 2. Greedily pick the ball covering the most uncovered words
    # But 729 balls is manageable. Let's do it.

    # Precompute all balls
    all_words = list(itertools.product(range(q), repeat=n))
    all_word_indices = {word_to_idx(w): w for w in all_words}

    # For efficiency, precompute for each word its neighbors (distance <= radius)
    # radius=1: neighbors are the word itself + all single-position changes
    def get_neighbors(word):
        """Return set of word indices within Hamming distance <= radius of word."""
        covered = set()
        covered.add(word_to_idx(word))
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word = list(word)
                    new_word[pos] = val
                    covered.add(word_to_idx(new_word))
        return covered

    # Precompute balls for all 729 words
    # This is 729 * (1 + n*(q-1)) = 729 * (1+6*2) = 729*13 = 9477 operations
    balls = {}
    for w in all_words:
        idx = word_to_idx(w)
        balls[idx] = get_neighbors(w)

    # Greedy set cover
    uncovered = set(range(total_words))
    code_indices = []

    while uncovered and elapsed() < DEADLINE:
        best_idx = None
        best_count = 0
        for idx in uncovered:
            # Try each uncovered word as a candidate center
            # But that's 729 * |uncovered| which is too slow if |uncovered| is large
            # Instead, for each candidate center, count how many uncovered words it covers
            # We need to be smarter.
            pass

        # Better approach: for each of the 729 possible centers, count uncovered coverage
        # 729 centers, each ball has 13 words -> 729*13 = 9477 checks per iteration
        # |uncovered| starts at 729, decreases each iteration
        # ~73 iterations * 9477 = ~690K operations. Fine.

        best_center = None
        best_cover = 0
        for center_idx in range(total_words):
            ball = balls[center_idx]
            count = 0
            for w_idx in ball:
                if w_idx in uncovered:
                    count += 1
            if count > best_cover:
                best_cover = count
                best_center = center_idx

        if best_center is None or best_cover == 0:
            break

        code_indices.append(best_center)
        ball = balls[best_center]
        for w_idx in ball:
            uncovered.discard(w_idx)

    # Convert to codewords
    code = [all_word_indices[i] for i in code_indices]

    if not code:
        code = [[0] * n]

    # --- Improvement phase: try to replace codewords to reduce count ---
    # For each codeword, check if removing it leaves all its covered words
    # still covered by other codewords.
    # Also try: for each codeword, see if replacing it with a different center
    # allows removing another codeword.

    # First pass: try removing codewords
    def is_covered(word_idx, code_indices_list, exclude_idx=None):
        """Check if word_idx is covered by some codeword in code_indices_list, excluding exclude_idx."""
        for ci, cidx in enumerate(code_indices_list):
            if ci == exclude_idx:
                continue
            if word_idx in balls[cidx]:
                return True
        return False

    # Try removing each codeword
    removed = []
    for i in range(len(code_indices)):
        if elapsed() > DEADLINE:
            break
        # Check if all words covered by code_indices[i] are still covered by others
        can_remove = True
        for w_idx in balls[code_indices[i]]:
            if not is_covered(w_idx, code_indices, i):
                can_remove = False
                break
        if can_remove:
            removed.append(i)

    # Remove from end to preserve indices
    for i in sorted(removed, reverse=True):
        code_indices.pop(i)

    # Second pass: try replacing pairs
    # For each pair (i, j), see if we can replace codeword i with some center c
    # such that removing codeword j is still valid
    # This is expensive. Let's do a simpler version:
    # For each codeword i, try replacing it with a nearby center that covers
    # more of the "hard" uncovered region.

    # Actually, let's try: for each codeword, try all 729 centers as replacement,
    # check if the resulting code still covers everything, and if it allows
    # removing another codeword.
    # 73 codewords * 729 centers * (check coverage + try removal) 
    # Check coverage: for each of 729 words, is it covered? -> 729 * 73 * 13 = ~690K per replacement check
    # 73 * 729 = 53K replacement checks * 690K = too slow.

    # Simpler: just do the removal pass again after replacement attempts.
    # Let's try a limited replacement: for each codeword, try the 13 neighbors
    # (distance 1) as replacement centers.

    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        for i in range(len(code_indices)):
            if elapsed() > DEADLINE:
                break
            # Try replacing codeword i with each neighbor
            current_center = code_indices[i]
            current_ball = balls[current_center]
            # Get neighbors of current center (distance 1)
            center_word = all_word_indices[current_center]
            for pos in range(n):
                orig_val = center_word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = list(center_word)
                    new_word[pos] = new_val
                    new_center = word_to_idx(new_word)
                    new_ball = balls[new_center]
                    # Check: does the new code (with i replaced by new_center) still cover everything?
                    # And can we remove some other codeword?
                    # First check full coverage
                    new_code_indices = code_indices[:]
                    new_code_indices[i] = new_center
                    # Check coverage
                    all_covered = True
                    for w_idx in range(total_words):
                        covered = False
                        for cidx in new_code_indices:
                            if w_idx in balls[cidx]:
                                covered = True
                                break
                        if not covered:
                            all_covered = False
                            break
                    if not all_covered:
                        continue
                    # Try to remove a codeword
                    for j in range(len(new_code_indices)):
                        if j == i:
                            continue
                        can_remove = True
                        for w_idx in balls[new_code_indices[j]]:
                            if not is_covered(w_idx, new_code_indices, j):
                                can_remove = False
                                break
                        if can_remove:
                            # Do the replacement and removal
                            code_indices = new_code_indices[:]
                            code_indices.pop(j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Final removal pass
    for _ in range(3):
        if elapsed() > DEADLINE:
            break
        removed = []
        for i in range(len(code_indices)):
            can_remove = True
            for w_idx in balls[code_indices[i]]:
                if not is_covered(w_idx, code_indices, i):
                    can_remove = False
                    break
            if can_remove:
                removed.append(i)
        if not removed:
            break
        for i in sorted(removed, reverse=True):
            code_indices.pop(i)

    code = [list(all_word_indices[i]) for i in code_indices]
    if not code:
        code = [[0] * n]

    return code
