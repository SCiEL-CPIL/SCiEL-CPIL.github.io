import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    if k == 3:
        res = _petersen_lift(g, deadline, start_time)
        if res is not None:
            return res
        return _luw(k, g, d=11)
    return _luw(k, g, d=7)


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _petersen_lift(g, deadline, start_time):
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges_base = 15

    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Coefficients of every closed non-backtracking walk of length 3..g-1.
    # A walk is a sequence of signed edges (eid, sign).
    def _closed_walk_coeffs(max_len):
        coeffs = set()
        for start_v in range(n_base):
            stack = []
            for next_v, e_id in base_adj[start_v]:
                u, v = pet_edges[e_id]
                sign = 1 if u == start_v else -1
                stack.append((next_v, e_id, 1, [(e_id, sign)]))
            while stack:
                curr_v, prev_e, length, path = stack.pop()
                if curr_v == start_v and length >= 3:
                    coeffs.add(tuple(path))
                if length >= max_len:
                    continue
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = pet_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    stack.append((next_v, e_id, length + 1, path + [(e_id, sign)]))
        return coeffs

    walk_paths = _closed_walk_coeffs(g)

    # Convert to sparse coefficient vectors (sum of signed edge counts).
    seen = set()
    walk_coeffs = []
    for path in walk_paths:
        coeff = [0] * n_edges_base
        for e_id, sign in path:
            coeff[e_id] += sign
        sparse = tuple((eid, c) for eid, c in enumerate(coeff) if c != 0)
        if sparse and sparse not in seen:
            seen.add(sparse)
            walk_coeffs.append(sparse)

    def _build_lift(volts, m):
        lift_edges = []
        for i, (u, v) in enumerate(pet_edges):
            vol = volts[i]
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + (t + vol) % m
                lift_edges.append((u_lift, v_lift))
        return lift_edges

    def _verify_girth(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != 3:
                return False
        for start in range(n):
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= girth_req:
                    continue
                if curr == start and dist >= 3:
                    return False
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        queue.append((nb, curr, dist + 1))
        return True

    # Deterministic linear-algebra solve over F_p for the smallest prime p >= 52.
    # Constraints: for each walk, sum c_e * v_e == 0 (mod p).
    # We fix the null-space dimension to 1 by pinning one free variable to 1.
    def _gauss_solve(rows, p):
        # rows: list of (row, rhs). Returns solution list or None.
        m = len(rows)
        n = n_edges_base
        # Build augmented matrix
        A = [[0] * (n + 1) for _ in range(m)]
        for i, (row, rhs) in enumerate(rows):
            for e, c in row:
                A[i][e] = c % p
            A[i][n] = rhs % p
        rank = 0
        col_of_pivot = []
        for col in range(n):
            if time.time() > deadline - 10.0:
                return None
            pivot = -1
            for r in range(rank, m):
                if A[r][col] % p != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
            A[rank], A[pivot] = A[pivot], A[rank]
            inv = pow(A[rank][col] % p, p - 2, p)
            for c in range(col, n + 1):
                A[rank][c] = (A[rank][c] * inv) % p
            for r in range(m):
                if r != rank and A[r][col] % p != 0:
                    factor = A[r][col]
                    for c in range(col, n + 1):
                        A[r][c] = (A[r][c] - factor * A[rank][c]) % p
            col_of_pivot.append((rank, col))
            rank += 1
        # Check consistency
        for r in range(rank, m):
            if A[r][n] % p != 0:
                return None
        # Free variables
        pivot_cols = set(c for _, c in col_of_pivot)
        free = [c for c in range(n) if c not in pivot_cols]
        if not free:
            return None
        # Pin the first free variable to 1, others to 0.
        pin = free[0]
        sol = [0] * n
        sol[pin] = 1
        for r, c in col_of_pivot:
            s = A[r][n]
            for cc in range(n):
                if cc != c:
                    s = (s - A[r][cc] * sol[cc]) % p
            sol[c] = s
        return sol

    best_n = float('inf')
    best_edges = None

    # Try primes in increasing order; also a few composites as fallback.
    primes = []
    m = 52
    while len(primes) < 40 and m < 400:
        if _is_prime(m):
            primes.append(m)
        m += 1

    for p in primes:
        if time.time() > deadline - 12.0:
            break
        candidate_n = n_base * p
        if candidate_n >= best_n:
            continue
        rows = [(tuple((e, c % p) for e, c in sparse), 0) for sparse in walk_coeffs]
        sol = _gauss_solve(rows, p)
        if sol is None:
            continue
        # Ensure at least one non-zero and gcd condition (connectivity).
        if all(x == 0 for x in sol):
            continue
        g_val = p
        for v in sol:
            g_val = math.gcd(g_val, v % p)
            if g_val == 1:
                break
        if g_val != 1:
            # Try a different pin (scale) to break the gcd.
            found = False
            for scale in range(2, p):
                g_val = p
                for v in sol:
                    g_val = math.gcd(g_val, (v * scale) % p)
                    if g_val == 1:
                        break
                if g_val == 1:
                    sol = [(v * scale) % p for v in sol]
                    found = True
                    break
            if not found:
                continue
        volts = [x % p for x in sol]
        lift_edges = _build_lift(volts, p)
        if _verify_girth(candidate_n, lift_edges, g):
            best_n = candidate_n
            best_edges = lift_edges
            break

    if best_edges is not None:
        return best_n, best_edges
    return None


def _luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, (g - 2) // 2)
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
