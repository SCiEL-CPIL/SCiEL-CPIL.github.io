# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 10.0


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


def _lps_graph(p):
    """
    Construct the LPS(2,p) Ramanujan graph: a 4-regular Cayley graph on SL(2,F_p).
    p must be a prime with p ≡ 3 (mod 4).
    Returns (n, edges).
    """
    assert p % 4 == 3, f"p={p} must be ≡ 3 mod 4"
    
    # Precompute modular inverses for 1..p-1
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Enumerate all elements of SL(2, F_p): [[a,b],[c,d]] with ad-bc = 1
    # Index each element
    elem_index = {}
    elems = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # Need ad - bc = 1 => d = (1 + bc) * a^{-1} if a != 0
                # If a == 0, then -bc = 1 => bc = -1, so c != 0 and b = -c^{-1}
                if a == 0:
                    if c == 0:
                        continue
                    # bc = -1 mod p => b = -inv[c]
                    b_val = (-inv[c]) % p
                    # But we're iterating b, so check if b == b_val
                    if b != b_val:
                        continue
                    d = 0  # a*d - b*c = 0 - b*c = -b*c = -(-1) = 1 ✓
                    elem = (a, b, c, d)
                else:
                    # d = (1 + b*c) * a^{-1} mod p
                    d = ((1 + b * c) * inv[a]) % p
                    elem = (a, b, c, d)
                
                if elem not in elem_index:
                    elem_index[elem] = len(elems)
                    elems.append(elem)
    
    n = len(elems)
    assert n == p * p * p, f"Expected {p**3} elements, got {n}"
    
    # Define the 4 generators (unit quaternions in SL(2,F_p)):
    # e1 = [[0, 1], [1, 0]]
    # e2 = [[0, -1], [1, 0]]  
    # e3 = [[1, 1], [-1, 1]]
    # e4 = [[1, -1], [1, 1]]
    # These satisfy e_i * e_i^T = I (they are symmetric up to sign)
    
    def multiply(x, y):
        """Matrix multiplication in SL(2, F_p)."""
        a1, b1, c1, d1 = x
        a2, b2, c2, d2 = y
        return ((a1*a2 + b1*c2) % p,
                (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p,
                (c1*b2 + d1*d2) % p)
    
    # The 4 generators
    gens = [
        (0, 1, 1, 0),       # e1
        (0, p-1, 1, 0),     # e2 = [[0,-1],[1,0]]
        (1, 1, p-1, 1),     # e3 = [[1,1],[-1,1]]
        (1, p-1, 1, 1),     # e4 = [[1,-1],[1,1]]
    ]
    
    # Build edges: for each element g, connect g to g*gen for each generator
    edges = set()
    for idx, g in enumerate(elems):
        for gen in gens:
            h = multiply(g, gen)
            h_idx = elem_index[h]
            edge = (min(idx, h_idx), max(idx, h_idx))
            edges.add(edge)
    
    return n, list(edges)


def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Girth check via non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True


def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges


def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # Strategy: Try LPS Ramanujan graphs for small primes p ≡ 3 (mod 4)
    # LPS(2,p) has p^3 vertices and girth ~ (4/3) log_p
    # For k=4, g=10: need girth >= 10
    # p=7: 343 vertices, expected girth ~ 10.7
    # p=11: 1331 vertices (larger, so try p=7 first)
    
    best_n = None
    best_edges = None
    
    # Try LPS graphs for primes p ≡ 3 (mod 4), starting from smallest
    if k == 4:
        # Primes ≡ 3 mod 4: 3, 7, 11, 19, 23, 31, 43, 47, ...
        candidate_primes = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71]
        for p in candidate_primes:
            if time.time() >= deadline:
                break
            if p % 4 != 3:
                continue
            try:
                n_lps, edges_lps = _lps_graph(p)
                if best_n is None or n_lps < best_n:
                    # Validate the graph
                    if _validate_graph(n_lps, edges_lps, k, g):
                        best_n = n_lps
                        best_edges = edges_lps
                        # If this is already very good (close to known best), stop
                        if n_lps <= 350:
                            break
            except Exception:
                continue
    
    # Fallback: LUW graph if LPS didn't work or for other k values
    if best_n is None:
        n_luw, edges_luw = _luw_graph(k, g)
        best_n = n_luw
        best_edges = edges_luw
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
