import time
from itertools import combinations

def _popcount(x):
    """Fast popcount using Python's int.bit_count()"""
    return x.bit_count()

def _is_regular_subset(adj_mask, subset_mask, n):
    """Check if the induced subgraph on subset_mask is regular."""
    k = _popcount(subset_mask)
    if k < 6:
        return True
    # For each vertex in subset, compute degree within subset
    first_deg = -1
    for i in range(n):
        if subset_mask & (1 << i):
            deg = _popcount(adj_mask[i] & subset_mask)
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
    return True

def _has_regular_subset_fast(adj_mask, n):
    """Check if any subset of size >= 6 induces a regular graph.
    Returns True if a regular subset exists (bad), False otherwise (good)."""
    # Only need to check subsets of size 6 (if no size-6 regular subset, 
    # larger ones are less likely but we should still check)
    # For efficiency, check size 6 first, then larger
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if _is_regular_subset(adj_mask, mask, n):
                return True
    return False

def _construct_algebraic_20():
    """Construct a 20-vertex graph using a specific permutation group structure.
    
    We use a graph based on the action of a transitive permutation group
    on 20 points. The idea: take a base graph and apply group symmetries
    in a way that breaks regularity of induced subgraphs.
    
    Strategy: Use a "blow-up" of a smaller base graph with irregular
    connection patterns, or use a specific known construction.
    
    Here we try a construction based on Z_20 with a specific edge set
    that is designed to avoid regular induced subgraphs.
    """
    n = 20
    edges = []
    adj_mask = [0] * n
    
    # Use a specific edge pattern: connect i to i+1, i+2, i+5, i+7 (mod 20)
    # This creates a circulant-like graph but with asymmetric steps
    steps = [1, 2, 5, 7]
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append([i, j])
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    return n, edges, adj_mask

def _construct_random_with_seed(n, seed):
    """Generate a random graph with a given seed."""
    import random
    rng = random.Random(seed)
    edges = []
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                edges.append([i, j])
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return n, edges, adj_mask

def _construct_paley_like(n):
    """Construct a Paley-like graph: connect i,j if (j-i) is a quadratic residue mod n.
    For n=20, use a modified version."""
    edges = []
    adj_mask = [0] * n
    
    # For n=20, define a specific set of "residues" to create asymmetry
    # Use a pseudo-random but deterministic pattern
    # Connect i to j if (j-i) mod 20 is in a specific set
    residues = {1, 2, 4, 8, 10, 12, 14, 16}  # Even numbers except 6, 18, 20
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues or (n - diff) in residues:
                edges.append([i, j])
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    return n, edges, adj_mask

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # First, try the algebraic 20-vertex construction
    n, edges, adj_mask = _construct_algebraic_20()
    if not _has_regular_subset_fast(adj_mask, n):
        return {'n': n, 'edges': edges}
    
    # Try Paley-like construction
    n, edges, adj_mask = _construct_paley_like(20)
    if not _has_regular_subset_fast(adj_mask, n):
        return {'n': n, 'edges': edges}
    
    # Try random constructions with different seeds for n=20
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        n, edges, adj_mask = _construct_random_with_seed(20, seed)
        if not _has_regular_subset_fast(adj_mask, n):
            return {'n': n, 'edges': edges}
    
    # Try n=19 with more attempts
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        n, edges, adj_mask = _construct_random_with_seed(19, seed)
        if not _has_regular_subset_fast(adj_mask, n):
            return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
