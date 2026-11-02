"""Greedy set-cover for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import heapq


def _ball_indices_fast(word, q, n, pows):
    """Fast ball: word as int index + offsets per position."""
    base = 0
    for i in range(n):
        base = base * q + word[i]
    ball = [base]
    for i in range(n):
        pw = pows[i]
        w_i = word[i]
        for v in range(q):
            if v != w_i:
                ball.append(base + (v - w_i) * pw)
    return ball


def _int_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (kill at 150s)

    total = q ** n
    pows = [q ** (n - 1 - i) for i in range(n)]

    # Precompute all balls for every word in the space
    all_balls = {}
    for idx in range(total):
        w = _int_to_word(idx, q, n)
        all_balls[idx] = _ball_indices_fast(w, q, n, pows)

    # --- Strong initial seed: direct sum of ternary tetracode (for q=3) ---
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed_words = [h + list(tail)
                      for h in tetracode
                      for tail in itertools.product(range(3), repeat=n - 4)]
        def to_idx(word):
            b = 0
            for x in word:
                b = b * q + x
            return b
        seed = [to_idx(w) for w in seed_words]
    else:
        seed = []

    # covered: set of indices
    covered = set()
    code_indices = list(seed)
    for idx in code_indices:
        for b in all_balls[idx]:
            covered.add(b)

    # uncovered set
    uncovered = set(range(total)) - covered

    # Greedy: repeatedly pick the candidate with max new coverage
    while uncovered:
        if time.time() > deadline:
            break
        # Find best candidate: maximize number of uncovered in its ball
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            ball = all_balls[idx]
            g = 0
            for b in ball:
                if b in uncovered:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_idx = idx
        if best_gain <= 0:
            break
        # Add best_idx as a codeword
        code_indices.append(best_idx)
        for b in all_balls[best_idx]:
            if b in uncovered:
                uncovered.discard(b)

    # --- Removal phase: try to remove redundant codewords ---
    # A codeword c is redundant if removing it doesn't leave any word uncovered
    # (i.e., every word in c's ball is covered by some other codeword)
    def is_redundant(c_idx, code_set):
        """Check if removing c_idx leaves all its ball covered by other codewords."""
        for b in all_balls[c_idx]:
            # b is covered by c; check if any other codeword covers b
            found = False
            for c2 in code_set:
                if c2 == c_idx:
                    continue
                if b in all_balls[c2]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Try removals in order of "least useful" codewords
    # A codeword is least useful if it has few unique covers (words only it covers)
    def unique_cover_count(c_idx, code_set):
        """Count words in c_idx's ball not covered by any other codeword."""
        count = 0
        for b in all_balls[c_idx]:
            covered_by_other = False
            for c2 in code_set:
                if c2 == c_idx:
                    continue
                if b in all_balls[c2]:
                    covered_by_other = True
                    break
            if not covered_by_other:
                count += 1
        return count

    # Sort by unique cover count (ascending) - try removing least unique first
    for _ in range(10):  # Multiple passes
        if time.time() > deadline:
            break
        code_set = set(code_indices)
        # Compute unique cover counts
        ucc = {}
        for c in code_indices:
            ucc[c] = unique_cover_count(c, code_set)
        # Try removing in order of increasing unique cover count
        sorted_codes = sorted(code_indices, key=lambda c: ucc[c])
        for c in sorted_codes:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
            if ucc[c] == 0:
                # This codeword has no unique covers, safe to remove
                code_set.discard(c)
                code_indices.remove(c)

    # Final greedy fill if removals left gaps
    covered_final = set()
    for idx in code_indices:
        for b in all_balls[idx]:
            covered_final.add(b)
    uncovered_final = set(range(total)) - covered_final
    while uncovered_final:
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            ball = all_balls[idx]
            g = 0
            for b in ball:
                if b in uncovered_final:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_idx = idx
        if best_gain <= 0:
            break
        code_indices.append(best_idx)
        for b in all_balls[best_idx]:
            if b in uncovered_final:
                uncovered_final.discard(b)

    # Convert back to words
    result = [_int_to_word(idx, q, n) for idx in code_indices]
    return result
# EVOLVE-BLOCK-END
