"""Covering code K_3(7,1) via bitset greedy + local refinement."""
import time
import random

_DEADLINE = 145.0  # safety margin before 150s kill


def construct(q, n, radius):
    t0 = time.time()
    # For this task: q=3, n=7, radius=1
    total = q ** n  # 2187
    # Precompute ball mask for each word: the word itself + all words at Hamming distance 1
    # Encode word as integer 0..total-1 in base q
    # Precompute neighbors: for each word, list of neighbors within distance 1
    
    # Build conversion: index <-> tuple
    def idx_to_word(i):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        return w  # little-endian: position 0 is least significant digit

    def word_to_idx(w):
        i = 0
        for pos in range(n - 1, -1, -1):
            i = i * q + w[pos]
        return i

    # Precompute ball masks
    # For each index, compute the bitmask of all words within Hamming distance 1
    ball_masks = [0] * total
    for i in range(total):
        w = idx_to_word(i)
        mask = 1 << i  # the word itself
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                j = word_to_idx(w)
                mask |= (1 << j)
            w[pos] = orig
        ball_masks[i] = mask

    # Greedy cover using bitsets
    def greedy_cover():
        uncovered = (1 << total) - 1
        selected = []
        while uncovered:
            if time.time() - t0 > _DEADLINE - 5:
                break
            # Find word whose ball covers most uncovered bits
            best_count = -1
            best_idx = -1
            # Iterate over all candidate words (could restrict to uncovered for speed, but balls overlap)
            # For efficiency, iterate all and count overlap
            # Optimization: only consider words that are themselves uncovered or cover uncovered
            for i in range(total):
                if time.time() - t0 > _DEADLINE - 5:
                    break
                overlap = ball_masks[i] & uncovered
                cnt = overlap.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    if cnt == 0:
                        break
            if best_idx < 0 or best_count <= 0:
                break
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return selected, uncovered

    # Try multiple random orderings / perturbations for diversity
    best_selected = None
    best_size = total  # worst case: all words

    # Run several greedy passes with slight randomization
    rng = random.Random(42)
    for trial in range(200):
        if time.time() - t0 > _DEADLINE - 3:
            break
        selected, uncovered = greedy_cover()
        if not uncovered:
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_selected = list(selected)
                # Local refinement: try to remove or swap
                if time.time() - t0 < _DEADLINE - 10:
                    selected = local_refine(selected, ball_masks, total, t0)
                    sz = len(selected)
                    if sz < best_size:
                        best_size = sz
                        best_selected = list(selected)

    if best_selected is None:
        # Fallback: direct sum construction (the seed)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return result

    # Convert selected indices back to word lists
    result = []
    for idx in best_selected:
        result.append(idx_to_word(idx))
    return result


def local_refine(selected, ball_masks, total, t0):
    """Try to reduce the code by removing redundant words or swapping."""
    selected_set = set(selected)
    # Compute coverage count for each word position
    # cov[i] = number of selected words whose ball covers word i
    cov = [0] * total
    for s in selected:
        bm = ball_masks[s]
        # For each bit set in bm, increment cov
        # Efficient: iterate over set bits
        x = bm
        while x:
            lb = x & (-x)
            i = lb.bit_length() - 1
            cov[i] += 1
            x ^= lb

    improved = True
    while improved and time.time() - t0 < _DEADLINE - 8:
        improved = False
        # Try removing each selected word
        for s in list(selected_set):
            if time.time() - t0 > _DEADLINE - 8:
                break
            # Check if removing s still covers everything
            # Word s is covered by other selected words if cov[s] >= 2
            # But we need ALL words to be covered. Removing s only affects words where cov was exactly 1 AND s was the sole cover.
            # A word w is uncovered after removing s iff cov[w] == 1 and w is in ball(s)
            # So we need: for all w in ball(s), cov[w] >= 2
            bm = ball_masks[s]
            x = bm
            safe = True
            while x:
                lb = x & (-x)
                i = lb.bit_length() - 1
                if cov[i] == 1:
                    safe = False
                    break
                x ^= lb
            if safe:
                # Remove s
                selected_set.discard(s)
                # Update cov
                x = bm
                while x:
                    lb = x & (-x)
                    i = lb.bit_length() - 1
                    cov[i] -= 1
                    x ^= lb
                improved = True
                break  # restart loop after removal

        if not improved:
            # Try swapping: remove s, add u if it covers more
            # This is expensive; do a limited pass
            if time.time() - t0 > _DEADLINE - 8:
                break
            # For a few random candidates, try swap
            candidates = list(selected_set)
            rng = random.Random(time.time_ns())
            rng.shuffle(candidates)
            for s in candidates[:10]:
                if time.time() - t0 > _DEADLINE - 8:
                    break
                bm_s = ball_masks[s]
                # Words that would become uncovered if we remove s
                # These are words in ball(s) with cov==1
                # We need a replacement u whose ball covers all those words
                # Find words with cov==1 in ball(s)
                critical = []
                x = bm_s
                while x:
                    lb = x & (-x)
                    i = lb.bit_length() - 1
                    if cov[i] == 1:
                        critical.append(i)
                    x ^= lb
                if not critical:
                    # Already handled above (would have been removed)
                    continue
                # Need u such that all critical words are in ball(u)
                # i.e., for all c in critical, u is within distance 1 of c
                # This means u must be in the intersection of balls of all critical words
                # Compute intersection mask
                inter = (1 << total) - 1
                for c in critical:
                    inter &= ball_masks[c]
                if inter == 0:
                    continue
                # Pick a u from inter that is not already selected and covers the most uncovered
                # Just pick the first available
                x = inter
                found = False
                while x:
                    lb = x & (-x)
                    u = lb.bit_length() - 1
                    if u not in selected_set:
                        # Do the swap
                        # Remove s
                        selected_set.discard(s)
                        xs = bm_s
                        while xs:
                            lbs = xs & (-xs)
                            i = lbs.bit_length() - 1
                            cov[i] -= 1
                            xs ^= lbs
                        # Add u
                        selected_set.add(u)
                        xu = ball_masks[u]
                        while xu:
                            lbu = xu & (-xu)
                            i = lbu.bit_length() - 1
                            cov[i] += 1
                            xu ^= lbu
                        improved = True
                        found = True
                        break
                    x ^= lb
                if found:
                    break

    return list(selected_set)
