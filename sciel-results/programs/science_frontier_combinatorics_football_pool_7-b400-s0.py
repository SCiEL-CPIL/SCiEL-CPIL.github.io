"""Ternary covering code K_3(7,1) — greedy + local search."""
import time
import numpy as np
import random


def _all_positions(q, n):
    # returns list of (coord, value) pairs for all single-coordinate changes
    pos = []
    for coord in range(n):
        for v in range(q):
            pos.append((coord, v))
    return pos


def _ball_indices(code, q, n, pos, base):
    # flat indices of all words within Hamming distance 1 of `code`
    # `code` is a list of ints; base[i] = sum_{j>i} q^j  (place value for coord i)
    out = [0] * (1 + n * (q - 1))
    idx = 0
    # center
    c = 0
    for i, v in enumerate(code):
        c = c * q + v
    out[idx] = c
    idx += 1
    for coord, v in pos:
        # compute index with coord set to v
        # base[coord] is place value for coord
        # original digit at coord:
        orig = code[coord]
        if orig == v:
            continue
        out[idx] = c + (v - orig) * base[coord]
        idx += 1
    return out[:idx]


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    total = q ** n

    # Precompute place values: base[i] = q^(n-1-i)
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute all single-coordinate change positions
    pos = _all_positions(q, n)
    npos = len(pos)

    # covered: boolean array of size total
    covered = np.zeros(total, dtype=bool)
    covered_count = 0
    code_list = []

    def add_code(code):
        nonlocal covered_count
        ball = _ball_indices(code, q, n, pos, base)
        idxs = np.array(ball, dtype=np.int64)
        mask = covered[idxs]
        new = int((~mask).sum())
        covered[idxs] = True
        covered_count += new
        code_list.append(code)

    # Greedy: pick word maximizing new coverage
    # For speed, we iterate over candidate words and compute new coverage via numpy
    # But iterating over all total words each step is expensive for large total.
    # Instead, we pick from uncovered words only, and use a heuristic:
    #   pick a random uncovered word, but bias toward high-degree.
    # Simpler: iterate a subset. For total up to ~2187, full scan is fine.

    # Precompute all words as a 2D array of shape (total, n)
    # For total=2187, n=7: 2187*7 = 15309 ints, fine.
    words = np.array([list(map(int, np.base_repr(i, q).zfill(n))) for i in range(total)], dtype=np.int64)
    # Precompute flat indices for all words (they are 0..total-1 in mixed-radix order? 
    # Actually base_repr with zfill gives most-significant first, matching our index formula.
    # So word i has flat index i. Good.

    # Precompute ball index arrays for every word? That's total * (1+n*(q-1)) ints.
    # For total=2187, ball size=1+7*2=15: 2187*15 = 32805 ints. Fine.
    balls = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
    for i in range(total):
        code = words[i].tolist()
        b = _ball_indices(code, q, n, pos, base)
        balls[i, :len(b)] = b
    ball_size = 1 + n * (q - 1)

    # Greedy loop
    remaining = total
    while remaining > 0:
        if time.time() > deadline - 30.0:
            break
        # For each candidate word, count how many of its ball points are uncovered
        # new_cov[i] = number of j in ball[i] where covered[j] is False
        # = ball_size - sum(covered[ball[i]])
        # Compute sum over balls using advanced indexing
        # covered[balls] gives shape (total, ball_size)
        cov_sum = covered[balls].sum(axis=1)
        new_cov = ball_size - cov_sum
        # Pick the word with max new_cov; tie-break by smallest index
        best_idx = int(np.argmax(new_cov))
        if new_cov[best_idx] <= 0:
            break
        add_code(words[best_idx].tolist())
        remaining = total - covered_count
        # If fully covered, stop
        if remaining <= 0:
            break

    # If not fully covered (due to time), we still have a partial code; 
    # but we need a VALID code. So we must ensure full coverage.
    # Force finish: keep adding until covered, even if slow.
    while remaining > 0:
        if time.time() > deadline - 5.0:
            # Emergency: add all uncovered words as codewords (valid but large)
            # Find uncovered words
            uncov = np.where(~covered)[0]
            for u in uncov:
                add_code(words[u].tolist())
            break
        cov_sum = covered[balls].sum(axis=1)
        new_cov = ball_size - cov_sum
        best_idx = int(np.argmax(new_cov))
        if new_cov[best_idx] <= 0:
            break
        add_code(words[best_idx].tolist())
        remaining = total - covered_count

    # Now we have a valid covering code. Try to reduce via local search.
    def is_valid():
        return covered_count == total

    # Local search: try deletions
    # For each codeword, check if removing it leaves all its ball points covered by others.
    # We maintain covered array; to test removal, we need to know if any ball point 
    # would become uncovered. We can compute, for each point, how many codewords cover it.
    # Use a count array.
    def build_counts():
        counts = np.zeros(total, dtype=np.int32)
        for code in code_list:
            i = int(np.dot(code, base))
            b = balls[i]
            counts[b] += 1
        return counts

    # Try deletions
    if time.time() < deadline - 10.0:
        counts = build_counts()
        # For each codeword, if every point in its ball has count >= 2, we can remove it.
        # Actually, we need: after removal, each point in its ball still has count >= 1.
        # i.e., for all p in ball[i], counts[p] >= 2.
        for ci in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            code = code_list[ci]
            i = int(np.dot(code, base))
            b = balls[i]
            if counts[b].min() >= 2:
                # Remove
                counts[b] -= 1
                code_list.pop(ci)
                # Update covered
                # covered[p] should remain True if counts[p] >= 1
                covered[b] = counts[b] >= 1
                covered_count = int(covered.sum())
                # Rebuild counts? No, we updated. But counts for other balls unchanged.
                # Continue.

    # Try swaps: remove A, add B, if coverage maintained and size doesn't increase.
    # This is expensive; do a limited random search.
    if time.time() < deadline - 5.0 and len(code_list) > 0:
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            if len(code_list) == 0:
                break
            ci = random.randrange(len(code_list))
            A = code_list[ci]
            iA = int(np.dot(A, base))
            bA = balls[iA]
            # Check if removing A leaves all bA covered by others
            # We need counts. Rebuild occasionally.
            # For simplicity, rebuild counts each time (expensive but bounded)
            counts = build_counts()
            if counts[bA].min() < 2:
                continue
            # Try adding a random B not already in code
            B = words[random.randrange(total)].tolist()
            if B in code_list:
                continue
            iB = int(np.dot(B, base))
            bB = balls[iB]
            # After removing A: counts' = counts - 1 on bA
            # After adding B: counts'' = counts' + 1 on bB
            # Need all points still covered: counts'' >= 1 everywhere.
            # Points affected: bA (decrement) and bB (increment).
            # Check: for p in bA, if counts[p] == 1 and p not in bB, then p becomes uncovered.
            # So: for all p in bA where counts[p]==1, p must be in bB.
            setB = set(bB.tolist())
            ok = True
            for p in bA:
                if counts[p] == 1 and p not in setB:
                    ok = False
                    break
            if not ok:
                continue
            # Also need: no point becomes uncovered elsewhere (only bA and bB change)
            # bB points: counts increase, so fine.
            # Do the swap
            code_list[ci] = B
            # Update covered
            covered[bA] = (counts[bA] - 1) >= 1  # after removing A
            # But we also add B: covered[bB] = True (they were already True or become True)
            covered[bB] = True
            covered_count = int(covered.sum())
            if covered_count != total:
                # Revert
                code_list[ci] = A
                covered[bA] = True
                covered[bB] = (counts[bB]) >= 1
                covered_count = int(covered.sum())

    # Final: ensure validity
    if covered_count != total:
        # Add all uncovered
        uncov = np.where(~covered)[0]
        for u in uncov:
            add_code(words[u].tolist())

    return [list(map(int, c)) for c in code_list]
