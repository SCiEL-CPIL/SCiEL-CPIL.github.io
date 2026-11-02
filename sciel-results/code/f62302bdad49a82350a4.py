"""Ternary covering code K_3(7,1): minimize codewords covering all 3^7 words within Hamming distance 1."""
import time
import random
import numpy as np


def _precompute_balls(q, n, radius):
    total = q ** n
    powq = [1] * n
    for i in range(1, n):
        powq[i] = powq[i - 1] * q
    digits = np.empty((total, n), dtype=np.int8)
    for i in range(total):
        x = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = x % q
            x //= q
    bsize = 1 + n * (q - 1)
    balls = np.empty((total, bsize), dtype=np.int32)
    for i in range(total):
        balls[i, 0] = i
        k = 1
        for pos in range(n):
            p = powq[pos]
            di = digits[i, pos]
            for d in range(q):
                if d != di:
                    balls[i, k] = i + (d - di) * p
                    k += 1
    return total, balls


def _cover_count(S, balls, total):
    return np.bincount(balls[S].ravel(), minlength=total)


def _removal_pass(S, balls, total, deadline):
    """Remove any codeword whose entire ball is covered by others; repeat to fixed point."""
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        cc = _cover_count(S, balls, total)
        unique = (cc[balls] == 1).sum(axis=1)
        rem = (unique == 0) & S
        if rem.any():
            S[rem] = False
            changed = True
    return S


def _reverse_delete(balls, total, deadline, rng, noise=0.0):
    S = np.ones(total, dtype=bool)
    while True:
        if time.time() > deadline:
            break
        cc = _cover_count(S, balls, total)
        unique = (cc[balls] == 1).sum(axis=1)
        rem = (unique == 0) & S
        if not rem.any():
            break
        red = (cc[balls[rem]] >= 2).sum(axis=1).astype(np.float64)
        if noise > 0.0:
            red = red + rng.random(len(red)) * noise
        idx = np.where(rem)[0][int(red.argmax())]
        S[idx] = False
    return S


def _forward_greedy(balls, total, deadline, rng, topk=1):
    covered = np.zeros(total, dtype=bool)
    code = []
    while not covered.all():
        if time.time() > deadline:
            break
        cnt = (balls[~covered]).sum(axis=1)
        if topk > 1:
            order = np.argsort(cnt)[::-1][:topk]
            c = int(order[rng.integers(len(order))])
        else:
            c = int(cnt.argmax())
        code.append(c)
        covered[balls[c]] = True
    S = np.zeros(total, dtype=bool)
    S[code] = True
    return S


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin under the 150 s kill

    # Generic fallback for non-task parameters (keep interface valid).
    if not (q == 3 and n == 7 and radius == 1):
        import itertools
        if q ** n <= 200000:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        return []

    total, balls = _precompute_balls(q, n, radius)
    rng = random.Random(12345)

    best = None
    best_size = total + 1

    def consider(S):
        nonlocal best, best_size
        S = _removal_pass(S.copy(), balls, total, deadline)
        size = int(S.sum())
        if size < best_size:
            best_size = size
            best = S.copy()

    # 1) Deterministic reverse-delete (strong baseline).
    consider(_reverse_delete(balls, total, deadline, rng, noise=0.0))

    # 2) Deterministic forward greedy + removal.
    consider(_forward_greedy(balls, total, deadline, rng, topk=1))

    # 3) Randomized restarts of both constructions until deadline.
    restart = 0
    while time.time() < deadline - 3.0:
        restart += 1
        r = random.Random(1000 + restart)
        if restart % 2 == 1:
            consider(_reverse_delete(balls, total, deadline, r, noise=1.0))
        else:
            consider(_forward_greedy(balls, total, deadline, r, topk=4))

    if best is None:
        best = np.ones(total, dtype=bool)

    code_idx = np.where(best)[0]
    # Decode indices to base-q digit lists.
    out = []
    for i in code_idx:
        x = int(i)
        word = [0] * n
        for pos in range(n - 1, -1, -1):
            word[pos] = x % q
            x //= q
        out.append(word)
    return out
