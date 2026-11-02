import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def _check_girth_nonbacktrack(n, adj, g):
    """
    Check girth >= g using non-backtracking BFS from vertex 0.
    For vertex-transitive graphs (circulants, Cayley), checking from 0 suffices.
    Returns True if no cycle shorter than g exists.
    """
    # State: (node, prev_node). dist = depth
    visited = set()
    queue = deque([(0, -1, 0)])
    visited.add((0, -1))
    
    while queue:
        u, p, d = queue.popleft()
        if d + 1 >= g:
            continue
        for v in adj[u]:
            if v == p:
                continue
            if v == 0 and d + 1 < g:
                return False
            state = (v, u)
            if state not in visited:
                visited.add(state)
                queue.append((v, u, d + 1))
    return True


def _circulant_girth_fast(n, a, h, g):
    """
    For C(n; {a, -a, h}) with h = n//2, compute girth analytically.
    The girth is the minimum cycle length. We find the smallest k >= 3
    such that there exists a closed non-backtracking walk of length k.
    For circulants, this can be checked by finding the smallest solution
    to certain congruences.
    
    A cycle of length k in C(n; {a, -a, h}) corresponds to a sequence of
    steps s_1, ..., s_k in {+a, -a, h} (mod n) that sum to 0 mod n,
    with no two consecutive steps being inverses (non-backtracking),
    and the partial sums being distinct (simple cycle).
    
    For our purposes, we just need to check if girth >= g.
    We do a BFS on the non-backtracking state space from vertex 0,
    but only track (current, prev) states, which is O(n * 3) states.
    """
    if n % 2 != 0:
        return False
    if math.gcd(n, a, h) != 1:
        return False
    if a == 0 or a == h or 2 * a % n == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u in range(n):
        adj[u].append((u + a) % n)
        adj[u].append((u - a) % n)
        adj[u].append((u + h) % n)
    
    return _check_girth_nonbacktrack(n, adj, g)


def _dihedral_girth_fast(n_rot, a, g):
    """
    Check Cay(D_{n_rot}, {s, r^a, r^-a}) for girth >= g.
    Total vertices = 2 * n_rot.
    """
    if math.gcd(n_rot, a) != 1:
        return False
    if 2 * a % n_rot == 0:
        return False
    
    total_n = 2 * n_rot
    adj = [[] for _ in range(total_n)]
    for i in range(n_rot):
        adj[i].append(n_rot + i)
        adj[i].append((i + a) % n_rot)
        adj[i].append((i - a) % n_rot)
    for i in range(n_rot):
        adj[n_rot + i].append((-i) % n_rot)
        adj[n_rot + i].append(n_rot + (i + a) % n_rot)
        adj[n_rot + i].append(n_rot + (i - a) % n_rot)
    
    return _check_girth_nonbacktrack(total_n, adj, g)


def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d = g - 4
    d = max(3, d)

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
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct(k, g):
    deadline = time.time() + 110
    
    if k == 3:
        # Moore bound for k=3, even g
        r = g // 2
        moore_bound = 2 * sum((k - 1) ** i for i in range(r))
        if moore_bound % 2 != 0:
            moore_bound += 1
        
        # Search for smallest n
        max_n_search = 2000
        
        for N in range(moore_bound, max_n_search + 1, 2):
            if time.time() > deadline - 5:
                break
                
            # Try Cyclic C(N; {a, N-a, N/2})
            h = N // 2
            for a in range(1, h):
                if math.gcd(h, a) != 1:
                    continue
                if _circulant_girth_fast(N, a, h, g):
                    edges = []
                    for u in range(N):
                        for v in ((u + a) % N, (u - a) % N, (u + h) % N):
                            if u < v:
                                edges.append((u, v))
                    return N, edges
                    
            # Try Dihedral
            n_rot = N // 2
            for a in range(1, n_rot):
                if _dihedral_girth_fast(n_rot, a, g):
                    edges = []
                    for i in range(n_rot):
                        u1 = i
                        v1 = n_rot + i
                        if u1 < v1: edges.append((u1, v1))
                        v2 = (i + a) % n_rot
                        if u1 < v2: edges.append((u1, v2))
                        v3 = (i - a) % n_rot
                        if u1 < v3: edges.append((u1, v3))
                        u2 = n_rot + i
                        v4 = (-i) % n_rot
                        if u2 < v4: edges.append((u2, v4))
                        v5 = n_rot + (i + a) % n_rot
                        if u2 < v5: edges.append((u2, v5))
                        v6 = n_rot + (i - a) % n_rot
                        if u2 < v6: edges.append((u2, v6))
                    return N, edges

    # Fallback to LUW
    try:
        return _luw_construct(k, g, deadline)
    except:
        return 0, []

# EVOLVE-BLOCK-END
