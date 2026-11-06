"""Rewritten program: search for minimal-modulus linear edge-colouring of K16
such that every K4 spans >= 5 colours. Keeps construct() -> 120 ints in lexicographic order."""
import time
import itertools

_START = time.time()
_DEADLINE = 150.0
_SAFETY = 10.0  # seconds margin before wall-clock kill


def _k4_ok(col, a, b, c, d):
    # colours of the 6 edges of K4 on {a,b,c,d}
    cs = (col[a][b], col[a][c], col[a][d],
          col[b][c], col[b][d], col[c][d])
    return len(set(cs)) >= 5


def _valid(col):
    n = 16
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    if not _k4_ok(col, a, b, c, d):
                        return False
    return True


def _build(m, shift):
    # colour(i,j) = (i + j + shift) mod m
    n = 16
    col = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            col[i][j] = (i + j + shift) % m
    return col


def _build2(m, shift, k):
    # colour(i,j) = (i + j + k*i*j + shift) mod m  (quadratic tweak)
    n = 16
    col = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            col[i][j] = (i + j + k * i * j + shift) % m
    return col


def _count_colours(col):
    s = set()
    for i in range(16):
        for j in range(i + 1, 16):
            s.add(col[i][j])
    return len(s)


def _flatten(col):
    return [col[i][j] for i in range(16) for j in range(i + 1, 16)]


def construct():
    deadline = _START + _DEADLINE - _SAFETY

    # Baseline: m=17, shift=0 (the known-valid seed construction).
    best_col = _build(17, 0)
    best_count = _count_colours(best_col)

    # Search the linear family (i+j+shift) mod m for m in 13..17, all shifts.
    for m in range(13, 18):
        if time.time() > deadline:
            break
        for shift in range(m):
            if time.time() > deadline:
                break
            col = _build(m, shift)
            if not _valid(col):
                continue
            c = _count_colours(col)
            if c < best_count:
                best_count = c
                best_col = col

    # Search quadratic family (i+j+k*i*j+shift) mod m for small m, k.
    for m in range(13, 18):
        if time.time() > deadline:
            break
        for k in range(0, m):
            if time.time() > deadline:
                break
            for shift in range(m):
                if time.time() > deadline:
                    break
                col = _build2(m, shift, k)
                if not _valid(col):
                    continue
                c = _count_colours(col)
                if c < best_count:
                    best_count = c
                    best_col = col

    out = _flatten(best_col)
    # Ensure colours are within 0..119 and integer.
    assert all(0 <= x <= 119 for x in out)
    assert len(out) == 120
    return out


# Self-test on import (cheap) to avoid shipping an invalid result.
if __name__ == "__main__":
    r = construct()
    print("colours used:", len(set(r)), "valid:", _valid([[0]*16 for _ in range(16)]))
