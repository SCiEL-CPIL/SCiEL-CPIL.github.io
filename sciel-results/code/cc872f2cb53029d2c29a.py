import time
import random
from itertools import combinations
from math import comb

START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # safety margin before 150s kill

def is_regular_subgraph(subset, adj_matrix, n):
    """Check if the induced subgraph on `subset` is regular.
    A regular subgraph means every vertex in the subset has the same degree
    within the subset (counting only edges to other vertices in the subset).
    """
    k = len(subset)
    if k < 2:
        # Single vertex: degree 0, trivially regular
        return True
    
    # Compute degrees of each vertex in the subset
    degrees = []
    for u in subset:
        deg = 0
        for v in subset:
            if u != v and adj_matrix[u][v]:
                deg += 1
        degrees.append(deg)
    
    # Check if all degrees are equal
    first_deg = degrees[0]
    for d in degrees[1:]:
        if d != first_deg:
            return False
    return True

def has_regular_subgraph_of_size_at_least_6(n, adj_matrix):
    """Check if there exists any subset of size >= 6 that induces a regular subgraph."""
    # We only need to check subsets of exactly size 6 (or more).
    # But checking all subsets of size >= 6 is expensive.
    # Key insight: if a subset of size m >= 6 induces a regular subgraph,
    # we can check subsets of size 6 first. But actually, a regular subgraph
    # of size m > 6 might exist without any size-6 regular subgraph within it.
    # However, the problem says "No set of 6 or more vertices may induce a regular subgraph."
    # So we need to check ALL subsets of size >= 6.
    # For efficiency, we'll check sizes 6, 7, ..., n.
    
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if is_regular_subgraph(subset, adj_matrix, n):
                return True
    return False

def validate_graph(n, edges):
    """Validate that the graph is simple and has no regular induced subgraph on >= 6 vertices."""
    # Build adjacency matrix
    adj_matrix = [[0] * n for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False  # no self-loops
        if not (0 <= u < n and 0 <= v < n):
            return False
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    
    # Check for duplicate edges
    edge_set = set()
    for u, v in edges:
        pair = (min(u, v), max(u, v))
        if pair in edge_set:
            return False  # duplicate edge
        edge_set.add(pair)
    
    # Check for regular subgraph of size >= 6
    if has_regular_subgraph_of_size_at_least_6(n, adj_matrix):
        return False
    
    return True

def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid_graph(n, edges):
        # Build adjacency bitsets
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check for regular induced subgraph of size >= 6
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                degs = []
                for u in subset:
                    deg = 0
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    deg = (adj[u] & mask).bit_count()
                    degs.append(deg)
                
                if len(set(degs)) == 1:
                    return False
        return True

    best_n = 5
    best_edges = []

    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5:
            break
            
        found = False
        for seed in range(200):
            if time.time() > deadline - 2:
                break
                
            random.seed(seed)
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            if is_valid_graph(n, edges):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
