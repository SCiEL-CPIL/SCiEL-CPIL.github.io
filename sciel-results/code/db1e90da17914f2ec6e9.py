import numpy as np
import itertools
import time
import random

Q = 3
N = 6
NUM_WORDS = Q ** N

all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Build ball data
ball_indices = []
ball_matrix = np.zeros((NUM_WORDS, NUM_WORDS), dtype=np.int8)
for i in range(NUM_WORDS):
    c = all_words[i]
    ball = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball.append(idx)
    ball_indices.append(np.array(ball, dtype=np.int32))
    ball_matrix[i, ball] = 1

# For each word w, precompute which codewords cover it (reverse ball)
covers_word = [np.where(ball_matrix[:, w] == 1)[0] for w in range(NUM_WORDS)]


def _tetracode():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code


def _cover_count(code, ball_indices):
    cc = np.zeros(NUM_WORDS, dtype=np.int32)
    for c in code:
        cc[ball_indices[c]] += 1
    return cc


def _try_reduce(code, ball_indices, rng):
    cc = _cover_count(code, ball_indices)
    changed = True
    while changed:
        changed = False
        order = list(range(len(code)))
        rng.shuffle(order)
        for idx in order:
            c = code[idx]
            if np.all(cc[ball_indices[c]] > 1):
                cc[ball_indices[c]] -= 1
                code.pop(idx)
                changed = True
                break
    return code, cc


def _local_search_1swap(code, ball_indices, covers_word, rng, iters=40):
    code = list(code)
    for _ in range(iters):
        if not code:
            break
        cc = _cover_count(code, ball_indices)
        idx = rng.randint(len(code))
        ci = code[idx]
        ball_i = ball_indices[ci]
        unique = ball_i[cc[ball_i] == 1]
        if len(unique) == 0:
            continue
        in_code = np.zeros(NUM_WORDS, dtype=bool)
        for c in code:
            in_code[c] = True
        candidates = set()
        for w in unique:
            candidates.update(covers_word[w].tolist())
        candidates -= set(code)
        if not candidates:
            continue
        for cj in list(candidates):
            ok = True
            for w in unique:
                if ball_matrix[cj, w] == 0:
                    ok = False
                    break
            if not ok:
                continue
            tc = list(code)
            tc[idx] = cj
            tc, _ = _try_reduce(tc, ball_indices, rng)
            if len(tc) < len(code):
                code = tc
                break
    return code


def construct(q, n, radius):
    if q is None or n is None or radius is None:
        return []
    if q != 3 or n != 6 or radius != 1:
        # general fallback greedy
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

    t0 = time.time()
    deadline = t0 + 140
    rng = random.Random()
    np_rng = np.random.RandomState()

    best_code = [tuple(w) for w in _tetracode()]
    best_size = len(best_code)

    # Precompute ball_indices for current params
    ball_idx = ball_indices
    bm = ball_matrix
    cw = covers_word

    while time.time() < deadline:
        # Greedy with randomized top-k
        covered = np.zeros(NUM_WORDS, dtype=bool)
        code = []
        while not np.all(covered):
            uncovered = ~covered
            scores = bm.dot(uncovered)
            mx = int(np.max(scores))
            if mx == 0:
                break
            thr = mx * 0.90
            cands = np.where(scores >= thr)[0]
            c = int(cands[np_rng.randint(len(cands))])
            code.append(c)
            covered[ball_idx[c]] = True

        # Reduce
        code, cc = _try_reduce(code, ball_idx, rng)

        # Local search: 1-for-1 swaps
        code = _local_search_1swap(code, ball_idx, cw, rng, iters=30)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            if best_size <= 73:
                # continue searching; try to improve
                pass

    # Return as list of lists
    return [list(all_words[c]) for c in best_code]
