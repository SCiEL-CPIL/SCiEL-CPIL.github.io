"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _tetracode_solution(q, n, radius):
    """Direct sum of ternary tetracode: 9 * q^(n-4) codewords."""
    if q != 3:
        return [[0] * n]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _ball_indices_vec(word_arr, q, n, radius):
    """Return a numpy int64 array of all word-indices within Hamming distance <= radius of word_arr."""
    # word_arr: 1-D numpy int array of length n
    if radius == 0:
        idx = int(np.dot(word_arr, q ** np.arange(n - 1, -1, -1)))
        return np.array([idx], dtype=np.int64)
    # radius 1: word + single-position changes
    # Precompute powers
    powq = q ** np.arange(n - 1, -1, -1)  # int64
    base_idx = int(np.dot(word_arr, powq))
    out = [base_idx]
    for pos in range(n):
        p = int(powq[pos])
        cur = word_arr[pos]
        for v in range(q):
            if v == cur:
                continue
            out.append(base_idx + (v - cur) * p)
    return np.array(out, dtype=np.int64)


def _build_ball_table(q, n, radius):
    """Precompute, for each position and value, the offset contribution. Returns a function to compute ball indices fast."""
    powq = q ** np.arange(n - 1, -1, -1)  # int64
    return powq


def _ball_indices_fast(word_arr, powq, q, n, radius):
    base_idx = int(np.dot(word_arr, powq))
    out = [base_idx]
    for pos in range(n):
        p = int(powq[pos])
        cur = int(word_arr[pos])
        for v in range(q):
            if v == cur:
                continue
            out.append(base_idx + (v - cur) * p)
    return np.array(out, dtype=np.int64)


def _verify(codewords, q, n, radius):
    """Check that every word in {0..q-1}^n is within radius of some codeword. Returns True/False."""
    total = q ** n
    powq = q ** np.arange(n - 1, -1, -1)
    covered = np.zeros(total, dtype=np.uint8)
    for w in codewords:
        wa = np.array(w, dtype=np.int64)
        ball = _ball_indices_fast(wa, powq, q, n, radius)
        covered[ball] = 1
    return bool(np.all(covered == 1))


def _local_search(codewords, q, n, radius, deadline, seed=0):
    """Local search: try to remove a codeword if coverage stays full, or swap (remove one, add another)."""
    rng = random.Random(seed)
    total = q ** n
    powq = q ** np.arange(n - 1, -1, -1)

    # Build coverage count array
    cov = np.zeros(total, dtype=np.int32)
    balls = []  # ball index array per codeword
    for w in codewords:
        wa = np.array(w, dtype=np.int64)
        b = _ball_indices_fast(wa, powq, q, n, radius)
        balls.append(b)
        cov[b] += 1

    # If not fully covered, cannot safely remove; but we assume initial is covered.
    if not bool(np.all(cov >= 1)):
        return codewords

    cur = list(codewords)
    cur_balls = balls
    cur_cov = cov
    best = len(cur)
    best_list = list(cur)

    def is_covered(c):
        return bool(np.all(c >= 1))

    # 1) Removal pass (vectorized-ish): for each codeword, check min(cov - its_ball)
    # We'll do iterative passes with random order.
    order = list(range(len(cur)))
    for it in range(200):
        if time.time() > deadline:
            break
        rng.shuffle(order)
        progress = False
        for i in order:
            if i >= len(cur):
                continue
            if time.time() > deadline:
                break
            # Check if removing i keeps coverage
            b = cur_balls[i]
            # temp: cov - 1 on b
            # We need to know if any element of cov[b] == 1
            if not bool(np.any(cur_cov[b] == 1)):
                # safe to remove
                cur_cov[b] -= 1
                del cur[i]
                del cur_balls[i]
                progress = True
                if len(cur) < best:
                    best = len(cur)
                    best_list = list(cur)
        if not progress:
            break

    # 2) Swap pass: remove one codeword (if it doesn't break coverage alone we can't; 
    #    but we can remove it temporarily if another added covers the gap). 
    #    Simpler: try remove i then add j; net effect. We approximate by:
    #    For each i, temporarily remove; if uncovered set U nonempty, find j whose ball covers U best.
    #    This is expensive; do a limited random sample.
    for it in range(30):
        if time.time() > deadline:
            break
        # sample a few i
        for _ in range(50):
            if time.time() > deadline:
                break
            i = rng.randrange(len(cur))
            b = cur_balls[i]
            # uncovered after removing i
            # U = indices where cur_cov==1 and in b
            U = b[cur_cov[b] == 1]
            if len(U) == 0:
                # pure removal
                cur_cov[b] -= 1
                del cur[i]
                del cur_balls[i]
                if len(cur) < best:
                    best = len(cur)
                    best_list = list(cur)
                continue
            # Try to find a replacement word j (not in cur) whose ball covers all of U (or most)
            # Candidate pool: random words
            found = False
            for _ in range(40):
                if time.time() > deadline:
                    break
                # random word
                cand = np.array([rng.randrange(q) for _ in range(n)], dtype=np.int64)
                cb = _ball_indices_fast(cand, powq, q, n, radius)
                # Does cb cover U?
                # check U subset of cb
                # use set intersection
                if len(U) <= 8:
                    cbset = set(cb.tolist())
                    if all(u in cbset for u in U.tolist()):
                        # do swap
                        cur_cov[b] -= 1
                        cur_cov[cb] += 1
                        cur[i] = cand.tolist()
                        cur_balls[i] = cb
                        if len(cur) < best:
                            best = len(cur)
                            best_list = list(cur)
                        found = True
                        break
            # if not found, no change

    return best_list


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    solution = _tetracode_solution(q, n, radius)

    # Verify initial (should be covered)
    if time.time() < deadline - 5.0:
        if not _verify(solution, q, n, radius):
            # fallback: brute force all words (only small cases)
            solution = [list(w) for w in itertools.product(range(q), repeat=n)]
            return solution

    # Local search with a couple of seeds
    best = solution
    for seed in (0, 1, 2):
        if time.time() > deadline - 10.0:
            break
        cand = _local_search(solution, q, n, radius, deadline - 5.0, seed=seed)
        if len(cand) < len(best):
            best = cand
        # Use improved as new starting point for next seed
        solution = best

    # Final safety verify
    if time.time() < deadline - 2.0:
        if not _verify(best, q, n, radius):
            return _tetracode_solution(q, n, radius)

    return best
# EVOLVE-BLOCK-END
