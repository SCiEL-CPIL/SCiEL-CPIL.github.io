import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _girth_ok(k, g, S, q):
    """Check girth >= g for circulant graph on Z_q with connection set S.
    Girth >= g iff no non-backtracking closed walk of length < g exists.
    In a Cayley graph, a non-backtracking closed walk of length t corresponds
    to a sequence s_1,...,s_t in S with sum = 0 mod q, s_i != -s_{i+1}.
    We use BFS over states (last_gen, partial_sum) tracking depth.
    """
    S_list = list(S)
    neg_map = {x: (-x) % q for x in S_list}
    k = len(S_list)
    
    # State: (last_gen_index, current_sum). Start: for each s in S, state (i, s) at depth 1.
    # At depth d (2..g-1), if current_sum == 0, we have a closed non-backtracking walk of length d.
    
    current = set()
    for i, s in enumerate(S_list):
        current.add((i, s % q))
    
    for depth in range(2, g):
        nxt = set()
        for last_i, s_val in current:
            neg_gen = neg_map[S_list[last_i]]
            for j, sj in enumerate(S_list):
                if sj == neg_gen:
                    continue
                ns = (s_val + sj) % q
                if ns == 0:
                    return False
                nxt.add((j, ns))
        if not nxt:
            break
        current = nxt
    
    return True


def _build_circulant(n, S):
    """Build edge list for circulant graph on Z_n with connection set S (symmetric, no 0)."""
    Sset = set(S)
    edges = []
    for i in range(n):
        for s in Sset:
            j = (i + s) % n
            if j > i:
                edges.append((i, j))
    return edges


def _search_circulant(k, g, deadline):
    """Search for smallest-n circulant k-regular graph with girth >= g."""
    best = None
    
    # Moore bound for odd girth g: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    moore = 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    n_start = max(moore, 428)
    n_end = n_start + 300
    
    for n in range(n_start, n_end + 1):
        if time.time() > deadline:
            break
        
        # For k=5: S = {a, b, c, -a, -b} where c = -a-b mod n
        # We need: all 5 distinct, none zero, and symmetric (which holds by construction)
        # Key: enumerate pairs (a, b) with a < b < n, c = -a-b mod n
        # To keep S symmetric: S = {a, b, c, -a, -b} and we need -c = a+b = -c mod n (always true)
        # So S is always symmetric. We just need 5 distinct nonzero elements.
        
        candidates = []
        # Strategy: try structured candidates first, then random-ish
        # 1. Near-uniform spacing: a ≈ n/5, b ≈ 2n/5
        base_a = n // 5
        base_b = (2 * n) // 5
        
        # Small perturbations around uniform
        for da in range(-8, 9):
            for db in range(-8, 9):
                a = (base_a + da) % n
                b = (base_b + db) % n
                c = (-a - b) % n
                S = {a, b, c, (-a) % n, (-b) % n}
                if len(S) == 5 and 0 not in S:
                    candidates.append((a, b, c))
        
        # 2. Powers of small bases
        for base in [2, 3, 5, 7, 11, 13, 17, 19]:
            a = pow(base, 1, n)
            b = pow(base, 2, n)
            c = pow(base, 3, n)
            S = {a, b, c, (-a) % n, (-b) % n}
            if len(S) == 5 and 0 not in S:
                candidates.append((a, b, c))
        
        # 3. Diverse structured candidates: a = i*n//5 + offset, b = j*n//5 + offset
        for i in range(1, 5):
            for j in range(1, 5):
                if i == j:
                    continue
                a0 = (i * n) // 5
                b0 = (j * n) // 5
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        a = (a0 + da) % n
                        b = (b0 + db) % n
                        c = (-a - b) % n
                        S = {a, b, c, (-a) % n, (-b) % n}
                        if len(S) == 5 and 0 not in S:
                            candidates.append((a, b, c))
        
        # 4. Random-like: use deterministic pseudo-random via hash
        import random
        rng = random.Random(n)
        for _ in range(60):
            a = rng.randint(1, n - 1)
            b = rng.randint(1, n - 1)
            c = (-a - b) % n
            S = {a, b, c, (-a) % n, (-b) % n}
            if len(S) == 5 and 0 not in S:
                candidates.append((a, b, c))
        
        # Try each candidate (limit to avoid timeout)
        seen = set()
        for (a, b, c) in candidates:
            if time.time() > deadline:
                break
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            S = [a, b, c, (-a) % n, (-b) % n]
            Sset = set(S)
            if len(Sset) != 5 or 0 in Sset:
                continue
            if _girth_ok(k, g, S, n):
                return (n, S)
    
    return best


def _luw_construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar / Laszlo bipartite graph (valid, larger).
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 100.0

    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []

    # Try circulant construction first
    result = _search_circulant(k, g, deadline)
    
    if result is not None:
        n, S = result
        edges = _build_circulant(n, S)
        return n, edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

# EVOLVE-BLOCK-END
