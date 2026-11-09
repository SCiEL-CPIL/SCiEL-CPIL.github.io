"""Football pool problem: ternary covering code K_3(6,1).
Simulated annealing initialized from the tetracode construction (81 words),
using a fast NumPy coverage model, with a wall-clock deadline guard."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 3.0
_RNG_SEED = 12345


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_code(q, n, radius):
    tet = _tetracode()
    words = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=n - 4):
            words.append(h + list(tail))
    return words


def _word_to_idx(word, q, n):
    idx = 0
    for pos in range(n):
        idx = idx * q + word[pos]
    return idx


def _ball_of(word, q, n, total):
    center = _word_to_idx(word, q, n)
    indices = [center]
    for pos in range(n):
        for val in range(q):
            if val == word[pos]:
                continue
            indices.append(center + (val - word[pos]) * (q ** pos))
    return indices


def _build_model(words, q, n):
    total = q ** n
    ball = np.zeros(total, dtype=np.int64)
    for w in words:
        for idx in _ball_of(w, q, n, total):
            ball[idx] += 1
    return ball


def _count_uncovered(ball):
    return int(np.count_nonzero(ball == 0))


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    rng = np.random.default_rng(_RNG_SEED)

    words = _initial_code(q, n, radius)
    total = q ** n
    ball = _build_model(words, q, n)

    best_words = [list(w) for w in words]
    best_len = len(words)
    best_uncovered = _count_uncovered(ball)

    def now():
        return time.time()

    T = 5.0
    cool = 0.9997
    min_T = 0.02

    while now() < deadline:
        if T < min_T:
            T = max(min_T, T * 1.3)

        i = int(rng.integers(0, len(words)))
        w = words[i]
        ball_list = _ball_of(w, q, n, total)

        newly = int(np.sum(ball[ball_list] == 1))
        cur_unc = _count_uncovered(ball)
        delta = newly - cur_unc

        if delta <= 0 or rng.random() < np.exp(-delta / T):
            for idx in ball_list:
                ball[idx] -= 1
            words.pop(i)
            cur_unc = _count_uncovered(ball)
            if cur_unc == 0 and len(words) < best_len:
                best_len = len(words)
                best_uncovered = 0
                best_words = [list(x) for x in words]
        T *= cool

    if best_uncovered != 0:
        return _initial_code(q, n, radius)
    return best_words
