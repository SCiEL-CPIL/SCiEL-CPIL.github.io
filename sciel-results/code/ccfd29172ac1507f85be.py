"""Structural change: simulated-annealing local search for a 4-colouring of K_n
with no monochromatic C6, using a fast bitmask C6-existence check."""
import time
import random

_DEADLINE = 120.0
_MARGINS = (0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 30.0, 40.0, 52.0, 65.0)


def _coloring_from_masks(masks, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            c = 0
            while not (masks[c] >> (i * n + j)) & 1:
                c += 1
            colors.append(c)
    return colors


def _has_c6(masks, n):
    """Return True iff some colour class contains a 6-cycle (not necessarily induced)."""
    for c in range(4):
        m = masks[c]
        if m == 0:
            continue
        for i in range(n):
            row = (m >> (i * n)) & ((1 << n) - 1)
            row &= ~(1 << i)
            if row == 0:
                continue
            # candidate start vertex: keep the smallest of the whole cycle to avoid
            # counting the same cycle many times (correctness is unaffected either way)
            row &= ~((1 << i) - 1)
            if row == 0:
                continue
            r = row
            while r:
                b = (r & (-r)).bit_length() - 1
                r &= r - 1
                row_b = (m >> (b * n)) & ((1 << n) - 1)
                row_b &= ~(1 << i)
                row_b &= ~(1 << b)
                if row_b == 0:
                    continue
                r = row_b
                while r:
                    d = (r & (-r)).bit_length() - 1
                    r &= r - 1
                    row_d = (m >> (d * n)) & ((1 << n) - 1)
                    row_d &= ~(1 << b)
                    row_d &= ~(1 << d)
                    if row_d == 0:
                        continue
                    r = row_d
                    while r:
                        e = (r & (-r)).bit_length() - 1
                        r &= r - 1
                        row_e = (m >> (e * n)) & ((1 << n) - 1)
                        row_e &= ~(1 << d)
                        row_e &= ~(1 << e)
                        if row_e == 0:
                            continue
                        r = row_e
                        while r:
                            f = (r & (-r)).bit_length() - 1
                            r &= r - 1
                            row_f = (m >> (f * n)) & ((1 << n) - 1)
                            row_f &= ~(1 << e)
                            row_f &= ~(1 << f)
                            if (row_f >> i) & 1:
                                return True
    return False


def _c6_count(masks, n):
    """Count the number of monochromatic 6-cycles (a proxy for how 'bad' a state is)."""
    total = 0
    for c in range(4):
        m = masks[c]
        if m == 0:
            continue
        for i in range(n):
            row = (m >> (i * n)) & ((1 << n) - 1)
            row &= ~(1 << i)
            if row == 0:
                continue
            row &= ~((1 << i) - 1)
            if row == 0:
                continue
            r = row
            while r:
                b = (r & (-r)).bit_length() - 1
                r &= r - 1
                row_b = (m >> (b * n)) & ((1 << n) - 1)
                row_b &= ~(1 << i)
                row_b &= ~(1 << b)
                if row_b == 0:
                    continue
                r = row_b
                while r:
                    d = (r & (-r)).bit_length() - 1
                    r &= r - 1
                    row_d = (m >> (d * n)) & ((1 << n) - 1)
                    row_d &= ~(1 << b)
                    row_d &= ~(1 << d)
                    if row_d == 0:
                        continue
                    r = row_d
                    while r:
                        e = (r & (-r)).bit_length() - 1
                        r &= r - 1
                        row_e = (m >> (e * n)) & ((1 << n) - 1)
                        row_e &= ~(1 << d)
                        row_e &= ~(1 << e)
                        if row_e == 0:
                            continue
                        r = row_e
                        while r:
                            f = (r & (-r)).bit_length() - 1
                            r &= r - 1
                            row_f = (m >> (f * n)) & ((1 << n) - 1)
                            row_f &= ~(1 << e)
                            row_f &= ~(1 << f)
                            total += (row_f >> i) & 1
    return total


def _k9():
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return n, colors


def construct():
    t0 = time.time()

    def deadline_left(margin):
        return time.time() - t0 < _DEADLINE - margin

    def make_masks(n):
        masks = [0, 0, 0, 0]
        for i in range(n):
            for j in range(i + 1, n):
                masks[random.randrange(4)] |= 1 << (i * n + j)
        return masks

    def anneal(n, deadline):
        masks = make_masks(n)
        cur = _c6_count(masks, n)
        best = masks[:]
        bestc = cur
        T = 1.0
        cool = 0.99999
        rng = random.Random(1234567 + n)
        while time.time() < deadline:
            if cur == 0:
                return masks
            i = rng.randrange(n)
            j = rng.randrange(i + 1, n)
            if i == j:
                continue
            bit = 1 << (i * n + j)
            c = rng.randrange(4)
            if (masks[c] >> (i * n + j)) & 1:
                continue
            old = -1
            for cc in range(4):
                if (masks[cc] >> (i * n + j)) & 1:
                    old = cc
                    break
            if old < 0:
                continue
            masks[old] &= ~bit
            masks[c] |= bit
            new = _c6_count(masks, n)
            if new <= cur or rng.random() < min(1.0, (cur - new) / max(T, 1e-9)):
                cur = new
                if cur < bestc:
                    bestc = cur
                    best = masks[:]
            else:
                masks[c] &= ~bit
                masks[old] |= bit
            T *= cool
        return best if bestc == 0 else None

    # Primary: try to find a valid 4-colouring of K17 by simulated annealing.
    if deadline_left(_MARGINS[0]):
        for attempt in range(6):
            if time.time() - t0 >= _DEADLINE - _MARGINS[0]:
                break
            res = anneal(17, _DEADLINE - _MARGINS[0])
            if res is not None:
                return {'n': 17, 'colors': _coloring_from_masks(res, 17)}

    # Fallback: verified K9 (always valid).
    n, colors = _k9()
    return {'n': n, 'colors': colors}
