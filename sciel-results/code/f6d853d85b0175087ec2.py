import numpy as np
import time
import random
import itertools

def _all_words(q, n):
    N = q ** n
    words = np.zeros((N, n), dtype=np.int8)
    for i in range(n):
        words[:, i] = (np.arange(N) // (q ** (n - 1 - i))) % q
    return words

def _build_cover_matrix(q, n):
    N = q ** n
    ball_size = 1 + n * (q - 1)
    words = _all_words(q, n)
    pow_q = [q ** (n - 1 - i) for i in range(n)]
    cover = np.zeros((N, ball_size), dtype=np.int32)
    for idx in range(N):
        w = words[idx]
        cover[idx, 0] = idx
        k = 1
        for i in range(n):
            for b in range(q):
                if b == w[i]:
                    continue
                cover[idx, k] = idx + (b - w[i]) * pow_q[i]
                k += 1
    return cover

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 140.0

    def _time_ok():
        return time.time() - start_time < DEADLINE

    N = q ** n
    ball_size = 1 + n * (q - 1)
    cover = _build_cover_matrix(q, n)
    words = _all_words(q, n)

    def _make_counts(cw_list):
        counts = np.zeros(N, dtype=np.int32)
        for c in cw_list:
            counts[cover[c]] += 1
        return counts

    def _try_remove(cw_list):
        new_list = list(cw_list)
        changed = True
        while changed and _time_ok():
            changed = False
            counts = _make_counts(new_list)
            for i, c in enumerate(new_list):
                if (counts[cover[c]] >= 2).all():
                    new_list.pop(i)
                    changed = True
                    break
        return new_list

    def _greedy_fill(cw_list):
        covered = np.zeros(N, dtype=bool)
        for c in cw_list:
            covered[cover[c]] = True
        uncovered_count = N - covered.sum()
        new_list = list(cw_list)
        while uncovered_count > 0 and _time_ok():
            n_uncovered = ball_size - covered[cover].sum(axis=1)
            best = int(np.argmax(n_uncovered))
            if n_uncovered[best] <= 0:
                break
            new_list.append(best)
            covered[cover[best]] = True
            uncovered_count = N - covered.sum()
        return new_list

    def _is_valid(cw_list):
        covered = np.zeros(N, dtype=bool)
        for c in cw_list:
            covered[cover[c]] = True
        return covered.all()

    # --- Initial seed ---
    seed_codewords = []
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                idx = 0
                for v in word:
                    idx = idx * 3 + v
                seed_codewords.append(idx)

    if not seed_codewords:
        covered = np.zeros(N, dtype=bool)
        seed_codewords = []
        while (N - covered.sum()) > 0 and _time_ok():
            n_uncovered = ball_size - covered[cover].sum(axis=1)
            best = int(np.argmax(n_uncovered))
            if n_uncovered[best] <= 0:
                break
            seed_codewords.append(best)
            covered[cover[best]] = True

    codewords = _try_remove(seed_codewords)
    best_codewords = list(codewords)
    best_size = len(codewords)

    rng = random.Random(42)

    # Shake and remove
    for _ in range(500):
        if not _time_ok():
            break
        perturbed = list(best_codewords)
        if perturbed:
            n_remove = min(len(perturbed), rng.randint(1, 3))
            for _ in range(n_remove):
                if perturbed:
                    perturbed.pop(rng.randrange(len(perturbed)))
        perturbed = _greedy_fill(perturbed)
        perturbed = _try_remove(perturbed)
        if len(perturbed) < best_size:
            best_size = len(perturbed)
            best_codewords = list(perturbed)

    # Single swaps
    for _ in range(200):
        if not _time_ok():
            break
        if not best_codewords:
            break
        trial = list(best_codewords)
        i = rng.randrange(len(trial))
        trial.pop(i)
        trial.append(rng.randrange(N))
        trial = _try_remove(trial)
        if _is_valid(trial) and len(trial) < best_size:
            best_size = len(trial)
            best_codewords = list(trial)

    result = [words[c].tolist() for c in best_codewords]
    return result
