"""Improved program using Cayley graph search and portfolio."""
import time
import sys
from collections import defaultdict, deque
import random

# EVOLVE-BLOCK-START

def _validate_graph(k, g, n, edges):
    """Validate that the graph is k-regular, simple, and has girth >= g."""
    # Check degree and simplicity
    degree = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        degree[u] += 1
        degree[v] += 1
    
    for d in degree:
        if d != k:
            return False
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth using BFS from each vertex (non-backtracking)
    for start in range(n):
        # BFS with state (node, prev) to avoid backtracking
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))  # (node, prev, dist)
        visited.add((start, -1))
        
        while queue:
            node, prev, dist = queue.popleft()
            if dist >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 < g:
                    return False  # Found cycle shorter than g
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    queue.append((nb, node, dist + 1))
    
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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

def _petersen_graph():
    """Petersen graph: k=3, g=5, n=10."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return 10, edges

def _heawood_graph():
    """Heawood graph: k=3, g=6, n=14."""
    lines = [
        [0, 1, 2], [0, 3, 4], [0, 5, 6],
        [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
    ]
    edges = []
    for i, line in enumerate(lines):
        line_node = 7 + i
        for p in line:
            edges.append((p, line_node))
    return 14, edges

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _find_cayley_graph(k, g, p):
    """
    Search for a subset S of F_p of size k such that the Cayley graph has girth >= g.
    Returns (p, edges) if found, else None.
    """
    if p < k + 1:
        return None
    
    # We fix 1 in S to break symmetry.
    # We need to choose k-1 other elements from 2..p-1.
    # This is a large search space.
    # We use a greedy/backtracking approach with pruning based on short cycles.
    
    # To speed up, we can check for small cycles (3, 4, 5, 6, 7, 8) as we build S.
    # A cycle of length L in Cayley graph corresponds to a sum of L elements from S U -S being 0.
    # Specifically, we need to avoid:
    # L=3: s1 + s2 + s3 = 0
    # L=4: s1 + s2 + s3 + s4 = 0
    # ...
    
    # Instead of full backtracking, we can try random subsets or structured ones.
    # Given time limit, we try a limited number of random candidates.
    
    candidates = []
    # Try symmetric sets? If S is symmetric, -S = S.
    # Then s1 + s2 + s3 = 0 is a triangle.
    
    # Let's just try random sets of size k.
    # We can optimize by checking constraints incrementally.
    
    def check_girth(S):
        # Check for cycles of length < g
        # BFS from 0
        dist = [-1] * p
        dist[0] = 0
        queue = deque([0])
        adj = defaultdict(list)
        # Build adjacency for 0? No, we need to check all nodes?
        # No, just check if any cycle exists.
        # We can just check sums.
        # But BFS is easier to implement correctly.
        
        # Actually, for a Cayley graph, we can just check if there are any zero-sum sequences.
        # Let's stick to BFS on the graph for the specific S.
        # But building the full graph is O(p*k).
        # BFS is O(p*k).
        # Total O(p*k) per candidate.
        
        # Build adj
        for u in range(p):
            for s in S:
                v = (u + s) % p
                adj[u].append(v)
        
        # BFS
        visited = set()
        queue = deque()
        queue.append((0, -1, 0))
        visited.add((0, -1))
        
        while queue:
            node, prev, d = queue.popleft()
            if d >= g - 1:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == 0 and d + 1 < g:
                    return False
                if (nb, node) not in visited:
                    visited.add((nb, node))
                    queue.append((nb, node, d + 1))
        return True

    # Random search
    # We try to find S.
    # We can bias towards "spread out" elements.
    
    # Try a few structured sets first?
    # E.g., geometric progression?
    
    # Let's do random search with a limit.
    # We have ~100s. p might be up to 1000.
    # k=5.
    # Number of candidates to try: maybe 1000?
    
    # To make it deterministic and fast, we can use a simple heuristic.
    # But random is safer for finding *a* solution.
    
    # We'll try 200 random sets.
    # If p is small, we can try more.
    
    max_attempts = 500
    for _ in range(max_attempts):
        # Generate random S
        # Must contain 1?
        # Let's just pick k random distinct non-zero elements.
        S = random.sample(range(1, p), k)
        if check_girth(S):
            # Construct edges
            edges = []
            for u in range(p):
                for s in S:
                    v = (u + s) % p
                    if u < v:
                        edges.append((u, v))
            return p, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Portfolio
    known_cages = {
        (3, 5): _petersen_graph,
        (3, 6): _heawood_graph,
    }
    
    if (k, g) in known_cages:
        try:
            n, edges = known_cages[(k, g)]()
            if _validate_graph(k, g, n, edges):
                return n, edges
        except Exception:
            pass

    best_n = float('inf')
    best_edges = None
    
    # 1. Try LUW
    try:
        n_luw, edges_luw = _luw_graph(k, g)
        if _validate_graph(k, g, n_luw, edges_luw):
            best_n = n_luw
            best_edges = edges_luw
    except Exception:
        pass
    
    # 2. Try Cayley graph search
    # Start with small primes
    # Moore bound for k=5, g=9 is 426.
    # So p must be at least 427?
    # Actually, n=p. So p >= 427.
    # Let's try primes from 431 upwards.
    
    # If LUW gave a very small graph, we might not need to search high.
    # But LUW for 5,9 is likely large.
    
    # Let's search primes in a range.
    # We can also try smaller primes if we relax girth? No, must be valid.
    
    # We'll try primes from 431 to 1000.
    # If we find one, it's likely smaller than LUW.
    
    current_prime = 431
    while current_prime < 2000 and time.time() < deadline:
        if _is_prime(current_prime):
            result = _find_cayley_graph(k, g, current_prime)
            if result:
                n_cay, edges_cay = result
                if n_cay < best_n:
                    best_n = n_cay
                    best_edges = edges_cay
                    # If we found a very small one, we can stop?
                    # Or keep searching for smaller?
                    # We are iterating increasing primes, so this is the smallest p found so far.
                    # We can break if we are confident this is good enough?
                    # But maybe a smaller p exists that we missed?
                    # We started from 431.
                    # So this is the smallest prime >= 431 that worked.
                    # We should break.
                    break
        current_prime += 1
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW if it was valid
    if best_n < float('inf'):
        return best_n, best_edges
        
    # If nothing worked, raise
    raise Exception("Failed to construct valid graph")

# EVOLVE-BLOCK-END
