import numpy as np
import itertools
import time
import math

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N
BALL_SIZE = 1 + N * (Q - 1)  # 13

all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
ball_indices = []

for i in range(NUM_WORDS):
    c = all_words[i]
    bi = [i]
    ball_matrix[i, i] = 1
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            nb = c.copy()
            nb[pos] = val
            idx = word_to_idx[tuple(nb)]
            bi.append(idx)
            ball_matrix[i, idx] = 1
    ball_indices.append(np.array(bi, dtype=np.int32))


def get_tetracode_code():
    """Fallback construction yielding 81 codewords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code


def _greedy_finish(covered, code, deadline):
    """Exact greedy to full coverage, returning (covered, code)."""
    uncovered = ~covered
    while uncovered.any():
        if time.time() > deadline:
            return covered, code
        scores = ball_matrix.dot(uncovered)
        c = int(np.argmax(scores))
        code.append(c)
        covered[ball_indices[c]] = True
        uncovered = ~covered
    return covered, code


def _vectorized_reduce(code, deadline, max_sweeps=2000):
    """Remove redundant codewords using a single coverage-count array."""
    cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        np.add.at(cov_counts, ball_indices[c], 1)

    sweeps = 0
    while sweeps < max_sweeps:
        if time.time() > deadline:
            break
        rem_mask = np.zeros(len(code), dtype=bool)
        for i in range(len(code)):
            if rem_mask[i]:
                continue
            c = code[i]
            if np.all(cov_counts[ball_indices[c]] > 1):
                rem_mask[i] = True
                cov_counts[ball_indices[c]] -= 1
        if not rem_mask.any():
            break
        code = [code[i] for i in range(len(code)) if not rem_mask[i]]
        sweeps += 1
    return code


def _smooth_greedy(covered, code, deadline, T=1.0):
    """Greedy with a smooth soft-max surrogate for candidate selection."""
    uncovered = ~covered
    while uncovered.any():
        if time.time() > deadline:
            return covered, code
        scores = ball_matrix.dot(uncovered).astype(np.float64)
        max_score = scores.max()
        # soft-max surrogate: equals max when T->0, rewards even coverage when T large
        shifted = (scores - max_score) / T
        # numerical stability: max of shifted is 0
        zmax = shifted.max()
        lse = zmax + np.log(np.exp(shifted - zmax).sum())
        soft = max_score - T * lse
        c = int(np.argmax(soft))
        code.append(c)
        covered[ball_indices[c]] = True
        uncovered = ~covered
    return covered, code


def _swap_improve(code, deadline, max_iters=500):
    """Simulated-annealing-style swap: remove one, add the best that keeps coverage."""
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if len(code) == 0:
            break
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            np.add.at(cov_counts, ball_indices[c], 1)
        # find a codeword whose removal leaves some uncovered
        candidates = []
        for i in range(len(code)):
            c = code[i]
            if np.any(cov_counts[ball_indices[c]] == 1):
                candidates.append(i)
        if not candidates:
            break
        i = candidates[np.random.randint(len(candidates))]
        c_remove = code[i]
        new_counts = cov_counts - ball_matrix[c_remove]
        uncovered = (new_counts == 0)
        if not uncovered.any():
            # removal alone keeps coverage -> just drop it
            code.pop(i)
            continue
        scores = ball_matrix.dot(uncovered)
        # choose a replacement
        repl = int(np.argmax(scores))
        code[i] = repl
        # re-add and reduce
        code = _vectorized_reduce(code, deadline, max_sweeps=50)
    return code


def _final_polish(code, deadline):
    """High-precision polish: exact greedy + exhaustive multi-removal."""
    # ensure full coverage
    covered = np.zeros(NUM_WORDS, dtype=bool)
    for c in code:
        covered[ball_indices[c]] = True
    if not covered.all():
        code, _ = _greedy_finish(covered, code, deadline)
    # aggressive reduction
    code = _vectorized_reduce(code, deadline, max_sweeps=5000)
    # try removing pairs
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        cov_counts = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code:
            np.add.at(cov_counts, ball_indices[c], 1)
        for i in range(len(code)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(code)):
                if time.time() > deadline:
                    break
                ci, cj = code[i], code[j]
                tmp = cov_counts - ball_matrix[ci] - ball_matrix[cj]
                if tmp.min() > 0:
                    cov_counts = tmp
                    code.pop(j)
                    code.pop(i)
                    changed = True
                    break
            if changed:
                break
    return code


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        bm = np.zeros((nw, nw), dtype=np.int8)
        for i in range(nw):
            c = all_w[i]
            bm[i, i] = 1
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    bm[i, wti[tuple(nb)]] = 1
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            scores = bm.dot(~covered)
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[bm[c] == 1] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s limit)

    best_code = get_tetracode_code()
    best_size = len(best_code)

    rng = np.random.default_rng(12345)

    for restart in range(100000):
        if time.time() > deadline:
            break

        # Smooth surrogate greedy with randomized temperature
        T = float(rng.choice([0.3, 0.7, 1.0, 1.5, 2.0]))
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        code, _ = _smooth_greedy(covered, code, deadline, T=T)

        # Vectorized reduction
        code = _vectorized_reduce(code, deadline, max_sweeps=200)

        # Swap improvement to escape local minima
        code = _swap_improve(code, deadline, max_iters=30)

        current_size = len(code)
        if current_size < best_size:
            best_size = current_size
            best_code = [list(all_words[c]) for c in code]
            if best_size <= 73:
                # found the best known; continue trying to improve
                pass

    # Final high-precision polish on the best solution
    if time.time() < deadline - 5:
        best_code_arr = [all_words.index(tuple(w)) if tuple(w) in word_to_idx else None for w in best_code]
        # build code as indices
        idx_code = []
        for w in best_code:
            idx_code.append(word_to_idx[tuple(w)])
        idx_code = _final_polish(idx_code, deadline)
        best_code = [list(all_words[c]) for c in idx_code]
        best_size = len(best_code)

    return best_code
