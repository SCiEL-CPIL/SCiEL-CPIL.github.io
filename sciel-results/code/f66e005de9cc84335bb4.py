"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START
def _has_regular_induced_6plus(n, adj):
    """Return True if any induced subgraph on >=6 vertices is regular."""
    # Precompute adjacency as bitmask for fast subset degree queries
    # adj[i] is an int bitmask of neighbors of i
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            # Check if induced subgraph is regular
            # Degree of vertex v within subset = popcount(adj[v] & subset_mask)
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            degs = []
            for v in subset:
                d = bin(adj[v] & subset_mask).count('1')
                degs.append(d)
            if len(set(degs)) == 1:
                return True
    return False

def _has_regular_induced_6plus_fast(n, adj):
    """Faster check using bitmask operations and early termination."""
    # For each subset of size >= 6, check if all vertices have same degree within subset
    # Use Gray code or just iterate combinations
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            first_deg = bin(adj[subset[0]] & subset_mask).count('1')
            regular = True
            for v in subset[1:]:
                d = bin(adj[v] & subset_mask).count('1')
                if d != first_deg:
                    regular = False
                    break
            if regular:
                return True
    return False

def _build_adj(edges, n):
    """Build adjacency bitmask list from edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_valid(n, adj):
    """Check if graph has no induced regular subgraph on >= 6 vertices."""
    return not _has_regular_induced_6plus_fast(n, adj)

def _random_graph(n, p, rng):
    """Generate random graph G(n, p)."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append([i, j])
    return edges

def _local_search(n, p_init, max_iters, rng, deadline):
    """Local search: start with random graph, flip edges to eliminate violations."""
    best_result = None
    best_score = -1
    
    for attempt in range(max_iters):
        if time.time() > deadline:
            break
        
        # Start with random graph
        edges = _random_graph(n, p_init, rng)
        adj = _build_adj(edges, n)
        
        # Iteratively fix violations
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find a violating subset
            for size in range(6, n + 1):
                found_violation = False
                for subset in itertools.combinations(range(n), size):
                    if time.time() > deadline:
                        return best_result
                    subset_mask = 0
                    for v in subset:
                        subset_mask |= (1 << v)
                    first_deg = bin(adj[subset[0]] & subset_mask).count('1')
                    regular = True
                    for v in subset[1:]:
                        d = bin(adj[v] & subset_mask).count('1')
                        if d != first_deg:
                            regular = False
                            break
                    if regular:
                        # Found a violation; try flipping an edge within or near this subset
                        # Pick a random edge to flip
                        u = rng.choice(subset)
                        v = rng.choice(subset)
                        if u == v:
                            continue
                        # Flip edge (u,v)
                        if adj[u] & (1 << v):
                            # Remove edge
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            edges = [e for e in edges if not (set(e) == {u, v})]
                        else:
                            # Add edge
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append([u, v])
                        improved = True
                        found_violation = True
                        break
                if found_violation:
                    break
            # Check if valid now
            if _is_valid(n, adj):
                if n > best_score:
                    best_score = n
                    best_result = {'n': n, 'edges': edges}
                break
        if best_result is not None and best_result['n'] >= n:
            break
    
    return best_result

def construct():
    """Construct a graph with no induced regular subgraph on >= 6 vertices, maximizing n."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    # Strategy: try increasing n from high to low
    # Known: 20-vertex graphs exist. Try to find one.
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        
        # Try different probabilities
        for p in [0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8]:
            if time.time() > deadline:
                break
            
            result = _local_search(n, p, max_iters=50, rng=rng, deadline=deadline)
            if result is not None and result['n'] >= n:
                return result
        
        # If we couldn't find one for this n, continue to smaller n
        # But we want to maximize n, so if n=20 fails, try harder or go to 19
    
    # Fallback: small graph that's definitely valid
    # For n < 6, no subset of size >= 6 exists, so any graph works
    return {'n': 5, 'edges': []}
# EVOLVE-BLOCK-END
