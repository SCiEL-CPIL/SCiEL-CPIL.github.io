import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    total_space = q ** n
    # Each codeword covers 1 + n*(q-1) = 17 points for q=3, n=8, radius=1
    covered_per_word = 1 + n * (q - 1)

    # Precompute all words as integers 0..total_space-1
    # Word i in base-q representation
    # Neighborhood: word itself + all words at Hamming distance 1

    # For speed, precompute the neighborhood mask for each possible codeword
    # Using Python integers as bitsets (6561 bits fits in ~1024 bytes)

    # Precompute neighbors for each word
    # neighbors[i] = bitmask of all words within distance <= radius of word i
    # For radius=1: word i + all words differing in exactly one position

    # Convert integer to tuple of digits
    def int_to_digits(x, n, q):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits

    def digits_to_int(digits, q):
        result = 0
        for i in range(len(digits) - 1, -1, -1):
            result = result * q + digits[i]
        return result

    # Precompute all neighborhood bitmasks
    # This is the expensive part: 6561 words * 17 neighbors each
    # But we can be smart: for each word, its neighbors are easy to compute

    # Actually, let's precompute the "neighborhood contribution" differently.
    # For each word w, the set of words it covers is {w} union {w with one digit changed}

    # Let's precompute for each position p and each word, what are the neighbors
    # when we change position p

    # More efficient: precompute neighbors for all 6561 words
    # neighbors_mask[w] = bitmask where bit i is 1 if word i is within radius of w

    # Precompute digit arrays
    all_digits = []
    for i in range(total_space):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        all_digits.append(d)

    # Precompute neighborhood masks
    # For radius=1: mask[w] has bits set for w and all w with one digit changed
    neighbor_masks = [0] * total_space
    for w in range(total_space):
        mask = 1 << w  # the word itself
        digits = all_digits[w]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                # Create new word
                new_digits = digits[:]
                new_digits[pos] = new_val
                new_w = digits_to_int(new_digits, q)
                mask |= (1 << new_w)
        neighbor_masks[w] = mask

    # Greedy set cover
    uncovered = (1 << total_space) - 1  # all bits set
    selected = []
    selected_set = set()

    # For greedy, at each step pick the word that covers the most uncovered points
    # To speed up, we can use a priority-based approach

    # Precompute: for each word, which other words are in its neighborhood
    # Actually, let's just do greedy with periodic re-evaluation

    # Initial greedy: pick words one by one, each time maximizing new coverage
    # This is O(n * total_space * bitops) which might be slow

    # Better: use a randomized/greedy hybrid
    # First, do a fast greedy with random restarts

    best_selected = None
    best_count = float('inf')

    num_restarts = 50

    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break

        random.seed(restart * 12345 + 7)
        uncovered = (1 << total_space) - 1
        selected = []
        selected_set = set()

        # Greedy: repeatedly pick the word covering most uncovered points
        # Use a heap or just scan (scanning 6561 words is fast with bit ops)

        while uncovered != 0:
            if time.time() > deadline - 25.0:
                break

            best_gain = -1
            best_word = -1
            # To speed up, sample a subset or use full scan
            # Full scan: for each candidate word not yet selected, compute gain
            # Gain = popcount(neighbor_masks[w] & uncovered)

            # Optimize: only consider words not in selected_set
            # But checking all 6561 each iteration is expensive
            # Let's try: maintain a list of candidates

            # For speed, do a partial scan or use random sampling + local improvement
            # Let's do: pick from a random subset of 200 candidates + the best from last round

            # Actually, let's just do full scan but with early termination
            # 6561 * bit_count on 6561-bit int... Python's int.bit_count() is fast

            # Let's sample: pick 500 random candidates + keep track of recent best
            candidates = []
            if len(selected_set) < total_space:
                # Get unselected words
                # This is expensive to compute each time
                # Instead, just sample from all and skip selected
                for _ in range(500):
                    w = random.randint(0, total_space - 1)
                    if w not in selected_set:
                        candidates.append(w)

            best_gain = -1
            best_word = -1
            for w in candidates:
                gain = (neighbor_masks[w] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word == -1 or best_gain <= 0:
                # Fallback: full scan
                for w in range(total_space):
                    if w in selected_set:
                        continue
                    gain = (neighbor_masks[w] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w

            if best_word == -1:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            uncovered &= ~neighbor_masks[best_word]

        # Local search: try to remove words
        # A word can be removed if all points it uniquely covers are covered by others
        # i.e., for all points p in neighbor_masks[w], there exists w' in selected\{w} with p in neighbor_masks[w']
        # Equivalent: neighbor_masks[w] & (uncovered_after_removal) == 0
        # where uncovered_after_removal = uncovered_before | (points only covered by w)
        # Simpler: remove w, check if still covered

        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try removing each word
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                w = selected[i]
                # Check if removing w still covers everything
                # Compute union of neighbors of all other selected words
                # If this union covers all of neighbor_masks[w]'s points that are in the total space... 
                # Actually, we need: for every point p that was covered by w, p is still covered by some other selected word
                # p is covered by w iff p in neighbor_masks[w]
                # p is covered by other iff exists w' in selected\{w} with p in neighbor_masks[w']
                # So: neighbor_masks[w] must be subset of (union of neighbor_masks[w'] for w' != w)
                # Equivalently: neighbor_masks[w] & ~(union_others) == 0

                # Compute union of all other neighbors
                # This is expensive: O(k * bitops) where k = len(selected)
                # Optimize: maintain a coverage count or use incremental approach

                # For now, do a direct check:
                # union_others = OR of neighbor_masks[w'] for all w' in selected, w' != w
                union_others = 0
                for j in range(len(selected)):
                    if j != i:
                        union_others |= neighbor_masks[selected[j]]

                if (neighbor_masks[w] & ~union_others) == 0:
                    # Can remove
                    selected.pop(i)
                    selected_set.discard(w)
                    improved = True
                    break  # restart removal loop

            # Try swaps: remove w, add w' if it helps
            # This is more complex; skip for now or do simple version

        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected[:]

        if best_count <= 403:  # Can't beat lower bound
            break

    # Convert selected word indices back to codeword lists
    if best_selected is None:
        # Fallback: use the tetracode product construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    result = []
    for w in best_selected:
        result.append(all_digits[w])

    # Ensure distinctness (should be by construction)
    return result
