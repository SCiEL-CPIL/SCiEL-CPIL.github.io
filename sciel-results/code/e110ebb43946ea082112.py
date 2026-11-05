import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Problem parameters
    N = q ** n  # 2187 for q=3, n=7
    ball_size = 1 + n * (q - 1)  # 15 for q=3,n=7

    # Precompute neighbors (ball of radius 1) for each word
    neighbors = np.zeros((N, ball_size), dtype=np.int32)

    # Helper to convert index to list of digits
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word

    # Precompute all words
    words = [idx_to_word(i) for i in range(N)]

    # Build word to index mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}

    # Compute neighbors
    for i in range(N):
        w_i = words[i]
        neighbors[i, 0] = i
        pos = 1
        for p in range(n):
            for val in range(q):
                if val == w_i[p]:
                    continue
                new_w = list(w_i)
                new_w[p] = val
                neighbors[i, pos] = word_to_idx[tuple(new_w)]
                pos += 1

    best_code = None
    best_size = float('inf')

    def get_cover_count(code):
        """Return cover_count array using numpy."""
        if not code:
            return np.zeros(N, dtype=np.int32)
        arr = np.array(code, dtype=np.int32)
        return np.bincount(neighbors[arr].ravel(), minlength=N).astype(np.int32)

    def is_covered(cover_count):
        """Check if all positions have cover >= 1."""
        return bool(np.all(cover_count >= 1))

    def prune_redundant(code, time_limit):
        """Remove redundant codewords using cover_count."""
        if not code:
            return code
        cover_count = get_cover_count(code)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            new_code = []
            for c in code:
                # Check if c is redundant: all its neighbors have cover_count > 1
                if np.all(cover_count[neighbors[c]] > 1):
                    cover_count[neighbors[c]] -= 1
                    changed = True
                else:
                    new_code.append(c)
            if changed:
                code = new_code
        return code

    def greedy_cover(uncovered_set, code, time_limit):
        """Greedy set cover using numpy for gain computation."""
        uncovered = np.array(uncovered_set, dtype=np.int32)
        if len(uncovered) == 0:
            return code
        # Build a boolean mask of uncovered
        unc_mask = np.zeros(N, dtype=bool)
        unc_mask[uncovered] = True

        while unc_mask.any() and time.time() < time_limit:
            # Compute gain for each candidate: number of uncovered neighbors
            gains = np.array([np.sum(unc_mask[neighbors[c]]) for c in range(N)], dtype=np.int32)
            max_gain = gains.max()
            if max_gain == 0:
                break
            best_candidates = np.where(gains == max_gain)[0]
            chosen = int(random.choice(best_candidates))
            code.append(chosen)
            unc_mask[neighbors[chosen]] = False

        return code

    def swap_search(code, time_limit):
        """Try to reduce code size by swapping codewords."""
        code = list(code)
        if len(code) <= 1:
            return code
        cover_count = get_cover_count(code)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each codeword and see if we can compensate
            for i in range(len(code)):
                if time.time() > time_limit:
                    break
                c_remove = code[i]
                # Check what would become uncovered if we remove c_remove
                test_cover = cover_count.copy()
                test_cover[neighbors[c_remove]] -= 1
                newly_uncovered = np.where(test_cover == 0)[0]
                if len(newly_uncovered) == 0:
                    # c_remove is redundant
                    cover_count = test_cover
                    code.pop(i)
                    improved = True
                    break
                # Try to find a single codeword that covers all newly_uncovered
                # and doesn't create new problems (i.e., it's not already in code,
                # or if it is, we'd need to also remove it - skip that case)
                # Find candidates: codewords whose ball covers all newly_uncovered
                found = False
                for j in range(len(code)):
                    if j == i:
                        continue
                    c_add = code[j]
                    # Check if c_add covers all newly_uncovered
                    add_ball = neighbors[c_add]
                    if np.all(np.isin(newly_uncovered, add_ball)):
                        # Adding c_add covers the gap, but c_add is already in code
                        # So removing c_remove and keeping c_add works
                        # But we need to verify no new uncovered is created elsewhere
                        # Since we're just removing c_remove and c_add stays,
                        # the only risk is words that were ONLY covered by c_remove
                        # which are exactly newly_uncovered, and c_add covers them
                        cover_count = test_cover
                        code.pop(i)
                        improved = True
                        found = True
                        break
                if not found:
                    # Try adding a new codeword not in code
                    code_set = set(code)
                    # Find best candidate: covers most newly_uncovered
                    # Use precomputed: for each position, which codewords cover it
                    # Simple approach: try all N positions
                    best_j = -1
                    best_cov = -1
                    for j in range(N):
                        if j in code_set:
                            continue
                        j_ball = neighbors[j]
                        cov = np.sum(np.isin(newly_uncovered, j_ball))
                        if cov == len(newly_uncovered) and cov > best_cov:
                            best_cov = cov
                            best_j = j
                    if best_j >= 0:
                        cover_count[neighbors[best_j]] += 1
                        cover_count[neighbors[c_remove]] -= 1
                        code[i] = best_j
                        improved = True
        return code

    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code
        code = list(code)
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)

        # Find uncovered using numpy
        cover_count = get_cover_count(code)
        uncovered = np.where(cover_count == 0)[0]

        # Repair
        code = greedy_cover(uncovered, code, time_limit)
        code = prune_redundant(code, time_limit)
        code = swap_search(code, time_limit)

        return code

    # Run multiple iterations with different strategies
    for iteration in range(200):
        if time.time() > deadline - 5:
            break

        # Start with greedy
        code = greedy_cover(set(range(N)), [], deadline - 3)
        code = prune_redundant(code, deadline - 3)
        code = swap_search(code, deadline - 3)

        # Try kicks
        for num_kicks in [1, 2, 3, 4]:
            if time.time() > deadline - 3:
                break
            code = kick_and_repair(code, num_kicks, deadline - 3)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

            if best_size <= 186:
                break

    # Convert internal indices back to lists of integers
    if best_code is None:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(words[c]) for c in best_code]
