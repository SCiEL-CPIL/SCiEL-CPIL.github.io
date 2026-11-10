"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_regular_induced(adj, subset):
    """Check if the induced subgraph on `subset` is regular."""
    n = len(subset)
    if n < 6:
        return False
    # Build adjacency for subset
    idx_map = {v: i for i, v in enumerate(subset)}
    degs = [0] * n
    for i, u in enumerate(subset):
        for j, v in enumerate(subset):
            if i != j and adj[u][v]:
                degs[i] += 1
    # Check if all degrees are equal
    return all(d == degs[0] for d in degs)

def _count_violations(adj, n):
    """Count number of vertex subsets of size >= 6 that induce a regular subgraph."""
    count = 0
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_induced(adj, subset):
                count += 1
    return count

def _find_violation(adj, n):
    """Find one violating subset (size >= 6) that induces a regular subgraph."""
    for size in range(6, n + 1):
        for subset in combinations(range(n), size):
            if _is_regular_induced(adj, subset):
                return subset
    return None

def _flip_edge(adj, u, v):
    """Flip edge between u and v."""
    adj[u][v] = not adj[u][v]
    adj[v][u] = not adj[v][u]

def _local_search(adj, n, deadline):
    """Iteratively flip edges to eliminate violations."""
    while time.time() < deadline:
        viol = _find_violation(adj, n)
        if viol is None:
            return True
        # Try flipping each edge within the violating subset
        found_fix = False
        random.shuffle(list(viol))
        for u in viol:
            for v in viol:
                if u < v:
                    # Save state
                    old = adj[u][v]
                    _flip_edge(adj, u, v)
                    # Check if this fixes the violation (and doesn't create new ones immediately)
                    new_viol = _find_violation(adj, n)
                    if new_viol is None:
                        # Great, we fixed it
                        found_fix = True
                        break
                    # Revert
                    _flip_edge(adj, u, v)
                    if found_fix:
                        break
            if found_fix:
                break
        if not found_fix:
            # Try flipping an edge outside the subset
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                old = adj[u][v]
                _flip_edge(adj, u, v)
                new_viol = _find_violation(adj, n)
                if new_viol is None:
                    found_fix = True
                    break
                _flip_edge(adj, u, v)
            if not found_fix:
                # Give up on this configuration
                return False
    return _find_violation(adj, n) is None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = {'n': 5, 'edges': []}
    
    # Try to build larger graphs
    for target_n in [10, 12, 14, 16, 18, 20]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            # Random graph with density ~0.5
            adj = [[False] * target_n for _ in range(target_n)]
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Local search to fix violations
            if _local_search(adj, target_n, deadline - 30):
                # Verify
                if _count_violations(adj, target_n) == 0:
                    edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i][j]]
                    best_result = {'n': target_n, 'edges': edges}
                    found = True
                    break
        
        if found:
            # Try to extend further
            continue
        else:
            # Couldn't find a valid graph of this size, try smaller
            break
    
    return best_result

# EVOLVE-BLOCK-END
