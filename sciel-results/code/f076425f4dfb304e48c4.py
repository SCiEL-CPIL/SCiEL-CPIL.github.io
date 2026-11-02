"""Rewritten program: greedy set-cover with redundant removal for K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import deque

# Fixed deadline with safety margin (evaluator kills at 150s; use 140s budget)
_DEADLINE = 140.0


def _ball_indices(q, n, radius):
    """Precompute for each index in [0, q**n) the set of indices within Hamming distance radius."""
    total = q ** n
    balls = [None] * total

    # Precompute all offsets (lists of (pos, delta) with len <= radius)
    # An offset is a tuple of (position, value_change) pairs
    # We'll generate all words within distance <= radius from 0, then use them as offsets

    # Generate all "delta words" of length n over {-1, 0, 1} with L1 norm <= radius
    # But for q=3, a change at position i can be +1 or -1 (mod q), so 2 choices per changed position.
    # Actually, for general q, there are (q-1) choices per changed position.

    # Let's generate all offset vectors: for each subset of positions of size <= radius,
    # and for each position in the subset, a nonzero delta in 1..q-1.
    offsets = [()]  # the zero offset
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for deltas in itertools.product(range(1, q), repeat=r):
                off = tuple(zip(positions, deltas))
                offsets.append(off)

    # Now for each base word, compute the ball
    # To speed up, we can iterate over all words and for each offset, compute the target index
    # But that's total * len(offsets) which for q=3, n=6, radius=1 is 729 * (1 + 6*2) = 729*13 = 9477. Fine.

    # Precompute index for a word given as tuple
    def word_to_idx(w):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Precompute the "add offset" operation on indices using digit decomposition
    # For efficiency, precompute for each offset the mapping from index to target index
    # Actually, let's just compute directly: for each base index, decompose to word, apply offset, recompute index.

    # Faster: precompute digits for each index
    digits = [[0] * n for _ in range(total)]
    for idx in range(total):
        temp = idx
        for i in range(n - 1, -1, -1):
            digits[idx][i] = temp % q
            temp //= q

    for idx in range(total):
        ball = set()
        for off in offsets:
            # Apply offset to digits[idx]
            new_digits = list(digits[idx])
            for pos, delta in off:
                new_digits[pos] = (new_digits[pos] + delta) % q
            # Convert back to index
            tidx = 0
            for i in range(n - 1, -1, -1):
                tidx = tidx * q + new_digits[i]
            ball.add(tidx)
        balls[idx] = ball

    return balls, digits


def _is_covered(balls, selected_set, uncovered):
    """Check if all uncovered words are covered by selected_set. Returns True if fully covered."""
    if not uncovered:
        return True
    for idx in selected_set:
        for w in balls[idx]:
            if w in uncovered:
                uncovered.discard(w)
                if not uncovered:
                    return True
    return not uncovered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    total = q ** n

    # Precompute balls (indices within Hamming distance <= radius)
    balls, digits = _ball_indices(q, n, radius)

    # Precompute: for each word, which potential codewords cover it (reverse mapping)
    # coverers[word_idx] = list of codeword indices that cover this word
    coverers = [[] for _ in range(total)]
    for cw in range(total):
        for w in balls[cw]:
            coverers[w].append(cw)

    best_solution = None
    best_count = total + 1

    # Greedy set cover with random restarts
    def greedy_random():
        """One greedy set-cover iteration with random tie-breaking."""
        uncovered = set(range(total))
        selected = []
        selected_set = set()

        while uncovered:
            # Find codeword that covers the most uncovered words
            best_cw = -1
            best_cover_count = 0
            candidates = []

            for cw in range(total):
                if cw in selected_set:
                    continue
                # Count how many uncovered words this cw covers
                cnt = 0
                for w in balls[cw]:
                    if w in uncovered:
                        cnt += 1
                if cnt > best_cover_count:
                    best_cover_count = cnt
                    best_cw = cw
                    candidates = [cw]
                elif cnt == best_cover_count and cnt > 0:
                    candidates.append(cw)

            if best_cover_count == 0:
                # Should not happen if balls are correct, but safety
                break

            # Randomly pick from candidates with max coverage
            chosen = random.choice(candidates)
            selected.append(chosen)
            selected_set.add(chosen)

            # Update uncovered
            for w in balls[chosen]:
                if w in uncovered:
                    uncovered.discard(w)

        # Remove redundant codewords
        # Try removing each codeword and check if coverage is still complete
        i = 0
        while i < len(selected):
            cw = selected[i]
            # Check if all words covered by cw are also covered by other selected codewords
            redundant = True
            for w in balls[cw]:
                # Check if w is covered by some other selected codeword
                covered_by_other = False
                for j in range(len(selected)):
                    if j == i:
                        continue
                    if w in balls[selected[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            if redundant:
                selected.pop(i)
            else:
                i += 1

        return selected

    # Initial solution: use the tetracode-based construction as a starting point
    # tetracode: 9 words of length 4, perfect code radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    def word_to_idx(w):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + w[i]
        return idx

    initial_solution = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_solution.append(word_to_idx(word))

    best_solution = initial_solution
    best_count = len(initial_solution)

    # Try greedy from scratch and from perturbed initial solutions
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break

        if restart == 0:
            # Greedy from scratch
            sol = greedy_random()
            if len(sol) < best_count:
                best_count = len(sol)
                best_solution = sol
        else:
            # Start from a perturbed version of the best solution
            # Randomly remove some codewords and re-greedy
            if best_solution is None:
                sol = greedy_random()
                if len(sol) < best_count:
                    best_count = len(sol)
                    best_solution = sol
                continue

            # Create a partial solution by removing some random codewords from best
            partial = list(best_solution)
            # Remove 10-30% of codewords randomly
            num_remove = max(1, int(len(partial) * random.uniform(0.1, 0.3)))
            to_remove = random.sample(partial, min(num_remove, len(partial)))
            partial_set = set(partial)
            for cw in to_remove:
                partial_set.discard(cw)
            partial = list(partial_set)

            # Now do greedy completion
            uncovered = set(range(total))
            for cw in partial:
                for w in balls[cw]:
                    if w in uncovered:
                        uncovered.discard(w)

            selected = list(partial)
            selected_set = set(partial)

            while uncovered:
                best_cw = -1
                best_cover_count = 0
                candidates = []

                for cw in range(total):
                    if cw in selected_set:
                        continue
                    cnt = 0
                    for w in balls[cw]:
                        if w in uncovered:
                            cnt += 1
                    if cnt > best_cover_count:
                        best_cover_count = cnt
                        best_cw = cw
                        candidates = [cw]
                    elif cnt == best_cover_count and cnt > 0:
                        candidates.append(cw)

                if best_cover_count == 0:
                    break

                chosen = random.choice(candidates)
                selected.append(chosen)
                selected_set.add(chosen)
                for w in balls[chosen]:
                    if w in uncovered:
                        uncovered.discard(w)

            # Remove redundant
            i = 0
            while i < len(selected):
                cw = selected[i]
                redundant = True
                for w in balls[cw]:
                    covered_by_other = False
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        if w in balls[selected[j]]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                if redundant:
                    selected.pop(i)
                else:
                    i += 1

            if len(selected) < best_count:
                best_count = len(selected)
                best_solution = selected

    # Convert best_solution indices back to words
    if best_solution is None:
        # Fallback: return all words (should not happen)
        return [digits[idx] for idx in range(total)]

    result = []
    for idx in best_solution:
        result.append(digits[idx])

    return result
# EVOLVE-BLOCK-END
