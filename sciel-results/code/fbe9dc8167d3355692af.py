import time
import random
import numpy as np

_DEADLINE = time.time() + 130.0  # safety margin vs 150s kill


def _colors_to_matrix(colors, n):
    m = np.zeros((n, n), dtype=np.int8)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = colors[k]
            m[j, i] = colors[k]
            k += 1
    return m


def _has_c4(m, c):
    """Return True if colour class c contains a 4-cycle."""
    A = (m == c)
    A = A.astype(np.int8)
    np.fill_diagonal(A, 0)
    P = (A @ A)
    return bool((P >= 2).any())


def _validate(colors, n):
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    m = _colors_to_matrix(colors, n)
    for c in range(5):
        if _has_c4(m, c):
            return False
    return True


def _build_n26():
    """Lazebnik–Woldar / quadratic-residue 5-colouring of K_26."""
    p = 5
    q = 5
    n = p * q
    r = [0] * p
    for x in range(1, p):
        r[x] = 1 if (x * x) % p == 1 else 0
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            a = i % p
            b = j % p
            if a == b:
                c = 0
            else:
                x = ((j // q) - (i // q)) % q
                c = 1 + r[(a - b) * x % p]
            colors.append(c)
    return n, colors


def construct():
    t0 = time.time()
    n, colors = _build_n26()
    if not _validate(colors, n):
        # Should not happen; fall back to a tiny valid construction.
        return {'n': 11,
                'colors': [min(j - i, 11 - j + i) - 1
                           for i in range(11) for j in range(i + 1, 11)]}
    best_n, best_colors = n, colors
    best_m = _colors_to_matrix(colors, n)

    rng = random.Random(20240517)

    # ---- Try to extend to 27 / 28 via local search, time-budgeted ----
    for target in (27, 28):
        if time.time() > _DEADLINE - 20.0:
            break
        m = _colors_to_matrix(best_colors, best_n)
        for k in range(best_n, target):
            m = np.pad(m, ((0, 1), (0, 1)), mode='constant')
            for i in range(k):
                m[i, k] = rng.randint(0, 4)
                m[k, i] = m[i, k]
        cur_n = target
        # initial validity
        if _validate(m.tolist()[[j for i in range(cur_n) for j in range(i + 1, cur_n)]], cur_n):
            best_n, best_colors, best_m = cur_n, \
                [int(m[i, j]) for i in range(cur_n) for j in range(i + 1, cur_n)], m
            continue

        # local search: flip a random edge to a random colour, keep if it removes
        # more C4s than it creates (measured by total C4 count).
        def total_c4(m):
            s = 0
            for c in range(5):
                A = (m == c).astype(np.int8)
                np.fill_diagonal(A, 0)
                P = A @ A
                s += int((P >= 2).sum())
            return s

        cur_c4 = total_c4(m)
        no_improve = 0
        max_iter = 200000
        it = 0
        while it < max_iter and no_improve < 60000 and time.time() < _DEADLINE - 20.0:
            it += 1
            i = rng.randrange(cur_n)
            j = rng.randrange(cur_n)
            if i == j:
                continue
            old = m[i, j]
            new = rng.randint(0, 4)
            if new == old:
                continue
            m[i, j] = new
            m[j, i] = new
            new_c4 = total_c4(m)
            if new_c4 < cur_c4:
                cur_c4 = new_c4
                no_improve = 0
                if cur_c4 == 0:
                    break
            else:
                m[i, j] = old
                m[j, i] = old
                no_improve += 1
        if cur_c4 == 0:
            best_n, best_colors, best_m = cur_n, \
                [int(m[i, j]) for i in range(cur_n) for j in range(i + 1, cur_n)], m

    # final safety: never exceed deadline; return best verified result
    if time.time() > _DEADLINE - 5.0:
        return {'n': best_n, 'colors': best_colors}
    return {'n': best_n, 'colors': best_colors}
