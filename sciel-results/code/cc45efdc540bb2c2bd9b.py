import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Try to construct a small graph using lifts of the Petersen graph
    # The Petersen graph has 10 vertices and girth 5.
    # A lift of size M has 10*M vertices.
    # We need girth >= g (16).
    # Expected girth ~ 5 + 2*log2(M).
    # For g=16, we need 5 + 2*log2(M) >= 16 => 2*log2(M) >= 11 => log2(M) >= 5.5 => M >= 45.
    # So we try M from 45 upwards.
    
    petersen_edges = _get_petersen_edges()
    
    best_n = float('inf')
    best_edges = None
    
    # Try increasing M
    # Moore bound for (3, 16) is 512. So 10*M must be >= 512 => M >= 52.
    # We start at M=52.
    for M in range(52, 200):
        if time.time() > deadline:
            break
            
        n = 10 * M
        # If we already have a better graph, skip larger n
        if n >= best_n:
            break
            
        # Try a few random lifts
        # Number of attempts depends on remaining time
        attempts = 20
        for _ in range(attempts):
            if time.time() > deadline:
                break
            edges = _random_lift(petersen_edges, M)
            if edges is None:
                continue
            if _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
                break # Found one for this M, try next M (smaller n)
                
    # If no lift worked, fall back to the seed's LUW construction
    if best_n == float('inf'):
        n_luw, edges_luw = _construct_luw(k, g)
        return n_luw, edges_luw
        
    return best_n, best_edges

def _get_petersen_edges():
    # Petersen graph: 10 vertices, 15 edges
    # Outer cycle 0-1-2-3-4-0
    # Inner star 5-7-9-6-8-5
    # Spokes i - (i+5)%10
    edges = []
    for i in range(5):
        edges.append((i, (i+1)%5))
    edges.append((4, 0)) # Already covered by loop? No, loop covers 0-1, 1-2, 2-3, 3-4. Need 4-0.
    # Actually, loop i in 0..4 adds (0,1), (1,2), (2,3), (3,4), (4,0). Correct.
    
    # Inner star
    # 5-7, 7-9, 9-6, 6-8, 8-5
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1)%5]))
        
    # Spokes
    for i in range(5):
        edges.append((i, i+5))
        
    return edges

def _random_lift(base_edges, M):
    """
    Create a random M-lift of the base graph.
    Base graph has 10 vertices.
    Lift has 10*M vertices.
    Vertex (u, i) in base becomes (u, i) in lift? No, we map base vertex u to 0..9.
    Lift vertex is (u, j) where j in 0..M-1.
    Edge (u, v) in base becomes edges ((u, j), (v, perm_uv[j])) for j in 0..M-1.
    """
    n_base = 10
    n_lift = n_base * M
    
    # Assign a random permutation to each directed edge of the base graph
    # Base edges are undirected, so (u, v) and (v, u) are the same edge.
    # We need a permutation for each direction?
    # If we use the same permutation for (u, v) and (v, u), the graph might not be connected or simple?
    # Actually, for a lift, we usually assign a permutation to each edge.
    # If the edge is undirected, we pick a permutation P.
    # Then edge (u, j) connects to (v, P(j)).
    # And edge (v, k) connects to (u, P^{-1}(k)).
    # This ensures consistency.
    
    # We need to map base edges to permutations.
    # Base edges: list of (u, v)
    # We create a dict: (u, v) -> perm
    # Since (u, v) is same as (v, u), we store for sorted tuple.
    
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            # Generate a random permutation of 0..M-1
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
            
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        # Determine direction
        # If u < v, then edge is (u, j) -> (v, p[j])
        # If u > v, then edge is (u, j) -> (v, p^{-1}[j])?
        # Actually, it's simpler:
        # For each j in 0..M-1:
        #   u_node = u * M + j
        #   v_node = v * M + p[j]  (if we define the lift this way)
        # But we must ensure that the edge is added only once.
        # The base edge (u, v) generates M edges in the lift.
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
            
    # Check for loops or multi-edges?
    # The base graph is simple. The lift of a simple graph by permutations is simple
    # provided that no permutation maps j to j for any j (which would create a loop in the lift? No, u!=v so u_node != v_node).
    # Multi-edges: Can (u, j) connect to (v, k) via two different base edges?
    # No, base graph is simple, so only one edge between u and v.
    # Can (u, j) connect to (v, k) via the same base edge but different j?
    # No, j is unique for each edge instance.
    # So the lift is simple.
    
    return lift_edges

def _has_girth_at_least(n, edges, g):
    """Check if graph has no cycle shorter than g. Uses BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        # BFS
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                else:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
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
