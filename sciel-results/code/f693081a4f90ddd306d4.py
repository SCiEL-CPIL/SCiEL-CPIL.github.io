"""Construct mutually orthogonal Latin squares of order 10."""
import time

# EVOLVE-BLOCK-START

_N = 10
_DEADLINE_MARGIN = 5.0


def _is_latin(square):
    n = len(square)
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [square[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True


def _orthogonal(a, b):
    n = len(a)
    seen = set()
    for i in range(n):
        ra = a[i]
        rb = b[i]
        for j in range(n):
            p = (ra[j], rb[j])
            if p in seen:
                return False
            seen.add(p)
    return True


def _generate_candidates():
    n = _N
    candidates = []

    # Group Z_2 x Z_5 squares.
    def elem_add(x, y):
        a1, b1 = x // 5, x % 5
        a2, b2 = y // 5, y % 5
        return ((a1 + a2) % 2) * 5 + ((b1 + b2) % 5)

    def elem_mul_scalar(x, k):
        result = 0
        for _ in range(k):
            result = elem_add(result, x)
        return result

    for g in range(1, 10):
        sq = [[elem_add(x, elem_mul_scalar(g, y)) for y in range(n)] for x in range(n)]
        if _is_latin(sq):
            candidates.append(sq)

    # Cyclic Z_10 squares.
    for a in range(1, 10):
        sq = [[(a*i + j) % n for j in range(n)] for i in range(n)]
        if _is_latin(sq):
            candidates.append(sq)

    # Base square permutations.
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    perms = [
        list(range(n)),
        list(reversed(range(n))),
        [i % 2 * 5 + i // 2 for i in range(n)],
        [i // 2 * 2 + i % 2 for i in range(n)],
    ]
    for rp in perms:
        for cp in perms:
            for sp in perms:
                sq = [[sp[base[rp[i]][cp[j]]] for j in range(n)] for i in range(n)]
                if _is_latin(sq):
                    candidates.append(sq)

    # Transposes.
    extra = []
    for sq in list(candidates):
        t = [[sq[i][j] for i in range(n)] for j in range(n)]
        if _is_latin(t):
            extra.append(t)
    candidates.extend(extra)

    # Deduplicate.
    unique = []
    seen_keys = set()
    for sq in candidates:
        key = tuple(tuple(row) for row in sq)
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(sq)

    return unique


def construct():
    t_start = time.time()
    deadline = t_start + 90.0 - _DEADLINE_MARGIN

    candidates = _generate_candidates()

    # Greedy selection.
    result = []
    for sq in candidates:
        if all(_orthogonal(sq, sel) for sel in result):
            result.append(sq)
        if len(result) == 9:
            break

    # Bounded backtracking to try to reach 9.
    if len(result) < 9 and time.time() < deadline:
        current = list(result)
        remaining = [sq for sq in candidates if sq not in current]
        remaining = remaining[:150]

        node_count = [0]
        max_nodes = 50000

        def backtrack(selected, pool):
            if time.time() > deadline or node_count[0] > max_nodes:
                return selected
            node_count[0] += 1
            if len(selected) == 9:
                return selected
            if not pool:
                return selected
            for i, sq in enumerate(pool):
                if all(_orthogonal(sq, sel) for sel in selected):
                    new_selected = selected + [sq]
                    new_pool = pool[:i] + pool[i+1:]
                    res = backtrack(new_selected, new_pool)
                    if len(res) == 9:
                        return res
            return selected

        bt = backtrack(current, remaining)
        if len(bt) > len(result):
            result = bt

    # Fallback.
    if not result:
        n = _N
        result = [[(i + j) % n for j in range(n)] for i in range(n)]

    # Verify and clean.
    verified = []
    for sq in result:
        if not _is_latin(sq):
            continue
        if all(_orthogonal(sq, v) for v in verified):
            verified.append(sq)

    if not verified:
        n = _N
        verified = [[(i + j) % n for j in range(n)] for i in range(n)]

    cleaned = []
    for sq in verified:
        cleaned_sq = [[int(sq[i][j]) for j in range(len(sq))] for i in range(len(sq))]
        cleaned.append(cleaned_sq)

    return cleaned

# EVOLVE-BLOCK-END
