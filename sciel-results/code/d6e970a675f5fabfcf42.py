"""Circulant graph construction for 5-regular graphs of girth >= 9."""
import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    
    For k=5, g=9: search for a circulant graph C(n, S) with girth >= 9.
    For other cases: use a general approach.
    """
    if k == 5 and g == 9:
        return _construct_5_9()
    else:
        return _construct_general(k, g)


def _construct_5_9():
    """
    Find a 5-regular circulant graph of girth >= 9 with as few vertices as possible.
    
    A circulant graph C(n, S) where S is a set of 5 elements in Z_n.
    The graph is 5-regular if we consider S as a connection set where each s gives
    edges i <-> (i+s) mod n. We need S to be symmetric (if s in S then -s in S) 
    for the graph to be undirected, OR we can use a set of 5 generators where
    the graph is defined by edges i -> (i+s) for s in S, and we ensure the graph
    is undirected by including both s and -s, but that would give degree 10.
    
    Actually, for a circulant graph to be 5-regular and undirected, we need:
    - Either S contains pairs {s, -s} plus one element s where s = -s (i.e., 2s = 0 mod n)
    - Or we use a different definition.
    
    Standard circulant graph C(n, S) where S is a symmetric subset of Z_n \ {0}:
    vertices are 0..n-1, edge between i and j iff (i-j) mod n in S.
    For 5-regular, |S| = 5 and S must be symmetric (s in S iff -s in S).
    This requires n to be even (so there's an element with 2s=0) and S = {a, -a, b, -b, n/2}.
    
    So we need n even, and S = {a, n-a, b, n-b, n/2} for some a, b.
    The girth is the length of the shortest cycle.
    
    Let's search for small even n >= 426 (Moore bound) and find a, b such that
    C(n, {a, n-a, b, n-b, n/2}) has girth >= 9.
    """
    start_time = time.time()
    time_limit = 100  # seconds
    
    # Moore bound for (5,9): 1 + 5*(1 + 4 + 16 + 64) = 426
    # Try n from 426 upward, even only
    for n in range(426, 2000, 2):
        if time.time() - start_time > time_limit:
            break
        
        half = n // 2
        
        # S = {a, n-a, b, n-b, n/2}
        # We need a != b, a != n-b, etc. (all distinct)
        # a in [1, half-1], b in [1, half-1], a != b, a != n-b (which is auto if a != b since n-b > half)
        
        # Try structured choices first
        found = False
        
        # Try a = 1, 2, 3, ... and b = 1, 2, 3, ...
        for a in range(1, half):
            if time.time() - start_time > time_limit:
                break
            for b in range(a + 1, half):
                # Check distinctness
                S = {a, n - a, b, n - b, half}
                if len(S) != 5:
                    continue
                
                # Check girth
                if _has_girth_at_least(n, S, 9):
                    # Build edges
                    edges = []
                    S_list = list(S)
                    for i in range(n):
                        for s in S_list:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        
        # Try some random choices
        rng = random.Random(n)
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            a = rng.randint(1, half - 1)
            b = rng.randint(1, half - 1)
            if a == b:
                continue
            S = {a, n - a, b, n - b, half}
            if len(S) != 5:
                continue
            if _has_girth_at_least(n, S, 9):
                edges = []
                S_list = list(S)
                for i in range(n):
                    for s in S_list:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: use a known construction
    return _construct_luw(5, 9)


def _has_girth_at_least(n, S, min_girth):
    """
    Check if the circulant graph C(n, S) has girth at least min_girth.
    Uses BFS from vertex 0 to find the shortest cycle.
    By symmetry, if the girth is < min_girth, there's a cycle through vertex 0.
    """
    S_list = list(S)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for s in S_list:
            j = (i + s) % n
            adj[i].add(j)
    
    # BFS from vertex 0
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    min_cycle = float('inf')
    
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                parent[nb] = node
                queue.append(nb)
            elif nb != parent[node] and node != parent[nb]:
                # Found a cycle
                # The cycle length is dist[node] + dist[nb] + 1
                # But we need to be careful: this gives a cycle that goes through node and nb
                cycle_len = dist[node] + dist[nb] + 1
                if cycle_len < min_cycle:
                    min_cycle = cycle_len
    
    # Also need to check cycles not through 0? 
    # Actually, by vertex-transitivity of circulant graphs, the girth is the same from any vertex.
    # So checking from vertex 0 is sufficient.
    
    return min_cycle >= min_girth


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback)."""
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


def _construct_general(k, g):
    """General construction for other k and g values."""
    return _construct_luw(k, g)
