"""Construct a k-regular graph with girth at least g using Lazebnik-Ustimenko-Woldar /
Hofmeister-style explicit constructions, choosing the smallest valid family found.

Strategy:
1. For the given (k, g), try a set of known explicit constructions that yield
   k-regular graphs with certified girth >= g.
2. Validate each candidate (regularity, simple, girth via non-backtracking BFS).
3. Return the smallest valid graph.
"""
import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Moore bound: lower bound on n
    if k == 1:
        moore = g
    elif k == 2:
        moore = g if g > 2 else 3
    else:
        # Moore bound for regular graphs: 1 + k + k(k-1) + ... + k(k-1)^(floor(g/2)-1)
        # For odd girth g=2r+1: 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # For even girth g=2r: 2 * sum_{i=0}^{r-1} k(k-1)^(i-1) ... 
        # Standard Moore bound:
        if g % 2 == 1:
            r = (g - 1) // 2
            moore = 1 + k * sum((k - 1)**i for i in range(r))
        else:
            r = g // 2
            moore = 2 * sum(k * (k - 1)**i for i in range(r - 1))
    
    best_n = float('inf')
    best_edges = None

    # Try Lazebnik-Ustimenko-Woldar (L-U-W) / Hofmeister construction
    # This gives a k-regular bipartite graph with girth >= d+5 for odd d
    # where the graph is defined over GF(q) with q = k (prime or 4)
    # Valid for prime k or k=4
    
    if k == 4 or _is_prime(k):
        edges = _luw_construct(k, g)
        if edges is not None:
            n = len(set(v for e in edges for v in e))
            if n < best_n:
                best_n = n
                best_edges = edges
                if time.time() > deadline:
                    break

    # Try complete bipartite-like constructions for small cases
    # For k=3, g=9: Petersen-like? No, Petersen has girth 5.
    # Actually for (3,9) the smallest is 22 vertices (Hoffman-Singleton is (7,5))
    # Let me try a few known small cages:
    # (3,9): 22 vertices - can construct via known methods
    # (5,9): 1116 vertices known
    
    # For the specific task (5,9), the L-U-W with q=5, d=4 gives:
    # girth >= d+5 = 9, and it's 5-regular bipartite
    # Number of vertices: 2 * 5^d = 2 * 625 = 1250? Let me check.
    
    # Actually the L-U-W graph CD(d,q) has 2*q^d vertices? No...
    # The construction: points are vectors in GF(q)^d, lines are vectors in GF(q)^d
    # Total vertices = 2 * q^d
    # For q=5, d=4: 2 * 625 = 1250 vertices
    # But the connected component of zero might be smaller.
    
    # The seed program returns the connected component, which for q=5,d=4
    # should be the full graph if it's connected.
    
    # Let's also try if we can get a better construction.
    # For (5,9), the known best is 1116. L-U-W gives 1250.
    # We need a better construction to beat 1250.
    
    # Try a different approach: use the fact that for (k, 2r+1) odd girth,
    # we can use finite geometry constructions.
    
    # For now, let's make sure the L-U-W construction is correct and fast.
    
    if time.time() > deadline:
        if best_edges is not None:
            return best_n, best_edges
        # Fallback: return a valid but large graph
        return _fallback_construct(k, g)
    
    if best_edges is None:
        return _fallback_construct(k, g)
    
    return best_n, best_edges


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


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=k.
    Returns list of edges or None if not valid."""
    q = k
    # Need girth >= g. Girth of CD(d,q) is at least d+5 for odd d.
    # So we need d+5 >= g, i.e., d >= g-5.
    # d must be odd for the standard girth guarantee.
    d = g - 5
    if d <= 2:
        d = 3
    if d % 2 == 0:
        d += 1  # make odd
    
    # Define field operations
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Rule for the construction
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Build adjacency
    # Vertices: (side, x) where side in {0, 1}, x is a tuple of length d+1 (first element is None)
    # Actually x is a tuple where x[0] is None and x[1..d] are in GF(q)
    
    # Start BFS from (0, (None,) + (0,)*d)
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    adj = {}
    
    for i, vertex in enumerate(order):
        adj[i] = []
        side, x = vertex
        for first in range(q):
            # Build the neighbor
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # x[c] = y[c] - y[a] * x[b] => x[c] = y[c] - x[a] * y[b] ... 
                    # Wait, let me re-derive.
                    # The incidence: for point p and line l, p is on l iff
                    # l[c] - p[c] = l[a_c] * p[b_c] for all c
                    # So if we're at a line l (side=1) and want to find point p (side=0):
                    # p[c] = l[c] - l[a_c] * p[b_c]
                    # This is implicit in p[b_c]. Let me use the seed's formula.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            adj[i].append(index[neighbor])
    
    # Build edge list
    edges = []
    for i, vertex in enumerate(order):
        if vertex[0] == 0:  # only add edges from point side to avoid duplicates
            for j in adj[i]:
                edges.append((i, j))
    
    n = len(order)
    
    # Validate: check regularity and girth
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != k for d in deg):
        return None
    
    # Check girth using non-backtracking BFS
    girth = _check_girth(n, edges, g)
    if girth < g:
        return None
    
    return edges


def _check_girth(n, edges, min_girth):
    """Check if the graph has girth at least min_girth using non-backtracking walks.
    Returns the actual girth found (or a large number if no short cycle)."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking BFS from each vertex
    # For each start vertex, do BFS on the state space (current, previous)
    # A cycle of length L is found if we return to the start vertex with a non-backtracking path of length L.
    
    best_girth = float('inf')
    
    for start in range(n):
        # BFS on non-backtracking paths starting from start
        # State: (current_node, previous_node, length)
        # We want to find the shortest cycle containing start
        visited = {}  # (current, previous) -> length
        queue = deque()
        
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        
        while queue:
            curr, prev, length = queue.popleft()
            
            if curr == start:
                # Found a cycle of length length
                if length < best_girth:
                    best_girth = length
                if best_girth >= min_girth:
                    return best_girth
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue  # non-backtracking
                if (nb, curr) in visited:
                    continue
                visited[(nb, curr)] = length + 1
                queue.append((nb, curr, length + 1))
        
        if best_girth >= min_girth:
            return best_girth
    
    return best_girth


def _fallback_construct(k, g):
    """Fallback: construct a valid k-regular graph with girth >= g.
    Uses a simple approach: take a large enough cycle and add edges carefully,
    or use a known small cage if available."""
    
    # For now, use the L-U-W construction with a larger d to ensure validity
    # even if the original d didn't work
    q = k
    if not (q == 4 or _is_prime(q)):
        # For composite k that isn't 4, we need a different approach
        # Use a complete bipartite graph K_{k,k} which has girth 4
        # This only works if g <= 4
        if g <= 4:
            edges = []
            for i in range(k):
                for j in range(k):
                    edges.append((i, j + k))
            return 2 * k, edges
        # Otherwise, we're in trouble. Return a large cycle-based construction.
        # This is a last resort.
        pass
    
    # Use L-U-W with larger d
    d = g - 5
    if d <= 2:
        d = 3
    if d % 2 == 0:
        d += 1
    # Try a few more d values
    for trial_d in range(d, d + 4, 2):
        edges = _luw_construct_with_d(k, trial_d)
        if edges is not None:
            n = len(set(v for e in edges for v in e))
            return n, edges
    
    # Ultimate fallback: return a valid but potentially large graph
    # Use a complete bipartite graph if g <= 4
    if g <= 4:
        edges = []
        for i in range(k):
            for j in range(k):
                edges.append((i, j + k))
        return 2 * k, edges
    
    # For larger g, we need a real construction. Return an error-like large graph.
    # This shouldn't happen for the task's parameters.
    raise ValueError(f"Could not construct a valid graph for k={k}, g={g}")


def _luw_construct_with_d(k, d):
    """L-U-W construction with a specific d value."""
    q = k
    if not (q == 4 or _is_prime(q)):
        return None
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    adj = {}
    
    for i, vertex in enumerate(order):
        adj[i] = []
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            adj[i].append(index[neighbor])
    
    edges = []
    for i, vertex in enumerate(order):
        if vertex[0] == 0:
            for j in adj[i]:
                edges.append((i, j))
    
    n = len(order)
    
    # Quick validation
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(dg != k for dg in deg):
        return None
    
    # Check girth
    girth = _check_girth(n, edges, g)
    if girth < g:
        return None
    
    return edges
