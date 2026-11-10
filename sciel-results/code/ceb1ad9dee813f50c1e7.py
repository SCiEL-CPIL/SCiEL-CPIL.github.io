"""Construct mutually orthogonal Latin squares of order 10."""

import time
import random
import sys

DEADLINE = 85.0  # safety margin below the 90s wall-clock limit


def _check_latin(sq):
    n = len(sq)
    for i in range(n):
        row = set(sq[i])
        if len(row) != n:
            return False
    for j in range(n):
        col = set(sq[i][j] for i in range(n))
        if len(col) != n:
            return False
    return True


def _check_orthogonal(a, b):
    seen = set()
    n = len(a)
    for i in range(n):
        for j in range(n):
            pair = (a[i][j], b[i][j])
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _verify(squares):
    if not squares:
        return False
    n = len(squares[0])
    for sq in squares:
        if len(sq) != n:
            return False
        for i in range(n):
            if len(sq[i]) != n:
                return False
            for v in sq[i]:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
        if not _check_latin(sq):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _check_orthogonal(squares[i], squares[j]):
                return False
    return True


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _make_latin_cyclic(n):
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def _try_find_orthogonal(base, n, max_iters=200000, rng=None):
    """Try to find a Latin square orthogonal to base via random permutation search."""
    if rng is None:
        rng = random.Random(42)

    symbols = list(range(n))

    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        diffs = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs)) == n:
            sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
            if _check_latin(sq) and _check_orthogonal(base, sq):
                return sq
    return None


def _try_find_second_orthogonal(s1, s2, n, max_iters=100000, rng=None):
    """Try to find a third square orthogonal to both s1 and s2."""
    if rng is None:
        rng = random.Random(123)

    p2 = [s2[0][j] for j in range(n)]
    symbols = list(range(n))

    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        diffs1 = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs1)) != n:
            continue
        diffs2 = [(p[j] - p2[j]) % n for j in range(n)]
        if len(set(diffs2)) != n:
            continue
        sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
        if _check_latin(sq) and _check_orthogonal(s1, sq) and _check_orthogonal(s2, sq):
            return sq
    return None


def _try_find_third_orthogonal(s1, s2, s3, n, max_iters=50000, rng=None):
    """Try to find a fourth square orthogonal to s1, s2, s3."""
    if rng is None:
        rng = random.Random(456)

    p2 = [s2[0][j] for j in range(n)]
    p3 = [s3[0][j] for j in range(n)]
    symbols = list(range(n))

    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        diffs1 = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs1)) != n:
            continue
        diffs2 = [(p[j] - p2[j]) % n for j in range(n)]
        if len(set(diffs2)) != n:
            continue
        diffs3 = [(p[j] - p3[j]) % n for j in range(n)]
        if len(set(diffs3)) != n:
            continue
        sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
        if _check_latin(sq):
            if _check_orthogonal(s1, sq) and _check_orthogonal(s2, sq) and _check_orthogonal(s3, sq):
                return sq
    return None


def _try_find_fourth_orthogonal(s1, s2, s3, s4, n, max_iters=30000, rng=None):
    """Try to find a fifth square orthogonal to s1, s2, s3, s4."""
    if rng is None:
        rng = random.Random(789)

    p2 = [s2[0][j] for j in range(n)]
    p3 = [s3[0][j] for j in range(n)]
    p4 = [s4[0][j] for j in range(n)]
    symbols = list(range(n))

    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        diffs1 = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs1)) != n:
            continue
        diffs2 = [(p[j] - p2[j]) % n for j in range(n)]
        if len(set(diffs2)) != n:
            continue
        diffs3 = [(p[j] - p3[j]) % n for j in range(n)]
        if len(set(diffs3)) != n:
            continue
        diffs4 = [(p[j] - p4[j]) % n for j in range(n)]
        if len(set(diffs4)) != n:
            continue
        sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
        if _check_latin(sq):
            if (_check_orthogonal(s1, sq) and _check_orthogonal(s2, sq) and
                    _check_orthogonal(s3, sq) and _check_orthogonal(s4, sq)):
                return sq
    return None


def _try_find_fifth_orthogonal(s1, s2, s3, s4, s5, n, max_iters=20000, rng=None):
    """Try to find a sixth square orthogonal to s1..s5."""
    if rng is None:
        rng = random.Random(1011)

    p2 = [s2[0][j] for j in range(n)]
    p3 = [s3[0][j] for j in range(n)]
    p4 = [s4[0][j] for j in range(n)]
    p5 = [s5[0][j] for j in range(n)]
    symbols = list(range(n))

    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        diffs1 = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs1)) != n:
            continue
        diffs2 = [(p[j] - p2[j]) % n for j in range(n)]
        if len(set(diffs2)) != n:
            continue
        diffs3 = [(p[j] - p3[j]) % n for j in range(n)]
        if len(set(diffs3)) != n:
            continue
        diffs4 = [(p[j] - p4[j]) % n for j in range(n)]
        if len(set(diffs4)) != n:
            continue
        diffs5 = [(p[j] - p5[j]) % n for j in range(n)]
        if len(set(diffs5)) != n:
            continue
        sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
        if _check_latin(sq):
            if (_check_orthogonal(s1, sq) and _check_orthogonal(s2, sq) and
                    _check_orthogonal(s3, sq) and _check_orthogonal(s4, sq) and
                    _check_orthogonal(s5, sq)):
                return sq
    return None


def construct():
    start_time = time.time()
    n = 10
    rng = random.Random(42)

    # First square: cyclic
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Try to find second square orthogonal to s1
    s2 = _try_find_orthogonal(s1, n, max_iters=500000, rng=rng)

    if s2 is None:
        if time.time() - start_time > DEADLINE:
            return [s1]
        return [s1]

    squares = [s1, s2]

    # Try to find third square
    if time.time() - start_time < DEADLINE - 2:
        s3 = _try_find_second_orthogonal(s1, s2, n, max_iters=300000, rng=rng)
        if s3 is not None:
            squares.append(s3)

    # Try to find fourth square
    if len(squares) >= 3 and time.time() - start_time < DEADLINE - 3:
        s4 = _try_find_third_orthogonal(s1, s2, s3, n, max_iters=200000, rng=rng)
        if s4 is not None:
            squares.append(s4)

    # Try to find fifth square
    if len(squares) >= 4 and time.time() - start_time < DEADLINE - 4:
        s5 = _try_find_fourth_orthogonal(s1, s2, s3, s4, n, max_iters=100000, rng=rng)
        if s5 is not None:
            squares.append(s5)

    # Try to find sixth square
    if len(squares) >= 5 and time.time() - start_time < DEADLINE - 5:
        s6 = _try_find_fifth_orthogonal(s1, s2, s3, s4, s5, n, max_iters=50000, rng=rng)
        if s6 is not None:
            squares.append(s6)

    # Verify all
    if _verify(squares):
        result = []
        for sq in squares:
            new_sq = []
            for row in sq:
                new_row = [int(v) for v in row]
                new_sq.append(new_row)
            result.append(new_sq)
        return result
    else:
        return [s1]
