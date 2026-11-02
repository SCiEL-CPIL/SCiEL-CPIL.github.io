import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        p = n
        d = 3
        while d * d <= p:
            if p % d == 0:
                break
            d += 2
        else:
            return p
        n += 2


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _inv(a, mod):
    # mod is prime
    return pow(a, mod - 2, mod)


def _girth_ok(k, g, S, q, S_neg_map, max_terms):
    # Girth >= g iff no non-backtracking closed walk of length < g exists.
    # In an abelian Cayley graph with 2-sided S, we track reachable sums
    # at each length. A sum of 0 at length >= 2 indicates a cycle.
    # Key insight: for non-backtracking walks, we only need to track the
    # current sum (not the last generator) because in an abelian group,
    # the set of reachable sums at length L determines all shorter/longer.
    # However, we DO need to avoid backtracking, so we track (last, sum).
    # Optimization: early exit as soon as 0 is found.
    target = g - 1
    S_list = list(S)
    nS = len(S_list)
    
    # Use a set of (last_gen_index, sum) pairs
    # At length 1: all single generators
    depth0 = set()
    for j, sj in enumerate(S_list):
        depth0.add((j, sj % q))
        if sj % q == 0:
            return False  # self-loop, but we already check 0 not in S
    
    for length in range(2, target + 1):
        seen = set()
        for last, s in depth0:
            li = S_list[last]
            neg = S_neg_map[li]
            for j, sj in enumerate(S_list):
                if sj == neg:
                    continue
                ns = (s + sj) % q
                seen.add((j, ns))
                if ns == 0:
                    return False
        depth0 = seen
        if not depth0:
            break
    return True


def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s (kill at 150s)

    # Moore bound: n >= 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i  (rough)
    # We just need n to be within [moore, max_vertices]; any valid k-regular
    # graph of girth>=g is automatically >= moore, so we just produce a valid one.
    if k < 3:
        # Degenerate: build a simple cycle (k=2) or empty (k=1)
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []

    # Choose prime power q: start at a prime >= some base, grow if needed.
    # For k=5, a prime q with good number-theoretic structure helps.
    # We'll try several q values and search S.
    # Base q must be >= k (so |S| can be k) and large enough to host a girth-g set.
    base_q = _next_prime(max(k, 7))

    candidates = []  # (n, S, q)
    
    # Generate more prime candidates with better distribution
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
    
    for trial in range(60):
        q = base_q + trial * 11
        if q % 2 == 0:
            q += 1
        if not _is_prime(q):
            continue

        # Need |S| = k, symmetric, no 0. For k odd, S = {c, a, b, -a, -b}.
        # For k even, S = {a, b, c, -a, -b, -c} (k=6) etc.
        if k % 2 == 1:
            # k = 2*m + 1
            m = (k - 1) // 2
        else:
            m = k // 2

        # Search over generator tuples. To keep it bounded, restrict search
        # to a structured family: use powers of a small base element.
        # Family: S = { r^0, r^1, ..., } combined with - and a shift.
        # Try S built from {x, y, z} where we vary x,y,z over a grid.
        # Grid over [1, q-1] is too big; use a coarse grid + field structure.

        # We'll pick candidate S from a parameterised family:
        #   even k: {a, b, c, -a, -b, -c}
        #   odd  k: {c, a, b, -a, -b}   (c = (a+b)//2 style or independent)
        # Use a larger grid with structured values
        import random
        small = [1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
        # Add q-based values
        small += [q-1, q-2, q-3, q-5, q-7, (q+1)//2, (q+3)//2, (q-1)//2, (q-3)//2]
        # Add some random values for diversity
        for _ in range(10):
            small.append(random.randint(1, q-1))
        small = [x % q for x in small if 0 < x % q < q]
        # dedupe, keep order
        seen_s = set()
        small_u = []
        for x in small:
            if x not in seen_s:
                seen_s.add(x)
                small_u.append(x)
        small_u = [x for x in small_u if x != 0]
        # Limit to manageable size
        if len(small_u) > 30:
            small_u = small_u[:30]

        found = False
        for a in small_u:
            if found:
                break
            for b in small_u:
                if b == a:
                    continue
                if found:
                    break
                for c in small_u:
                    if c == a or c == b:
                        continue
                    if k % 2 == 1:
                        # S = {c, a, b, -a, -b}
                        S = [c, a, b, (-a) % q, (-b) % q]
                    else:
                        S = [a, b, c, (-a) % q, (-b) % q, (-c) % q]
                    # S must have exactly k distinct elements, none 0
                    if len(set(S)) != k or 0 in S:
                        continue
                    # 2-sided: S == -S
                    Sset = set(S)
                    if Sset != {(-x) % q for x in Sset}:
                        continue
                    # girth check
                    S_neg_map = {x: (-x) % q for x in Sset}
                    if _girth_ok(k, g, S, q, S_neg_map, 0):
                        candidates.append((q, S))
                        found = True
                        break
        if found:
            break
        if time.time() > deadline - 30.0:
            break

    if not candidates:
        # Fallback: LUW-style construction (guaranteed valid, larger).
        return _luw_construct(k, g)

    # Pick the candidate with smallest q (=> smallest n)
    candidates.sort(key=lambda t: t[0])
    q, S = candidates[0]
    n = q
    S_list = list(S)
    edges = []
    Sset = set(S_list)
    for i in range(n):
        for s in S_list:
            j = (i + s) % q
            if j > i:
                edges.append((i, j))
    return n, edges


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

# EVOLVE-BLOCK-END
