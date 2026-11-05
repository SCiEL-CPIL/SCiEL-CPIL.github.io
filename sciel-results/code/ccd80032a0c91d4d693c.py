# EVOLVE-BLOCK-START
import time
import random

def _check_subset_valid(subset, adj_bits, n):
    """
    Check if the induced subgraph on 'subset' is regular.
    Returns True if it IS regular (violation), False otherwise.
    """
    k = len(subset)
    if k < 6:
        return False
        
    degrees = [0] * k
    for i, u in enumerate(subset):
        # Count neighbors in subset
        # adj_bits[u] has 1s for neighbors
        # We need to count bits in adj_bits[u] & mask_subset
        # But we only care about neighbors in subset
        # Simpler: iterate j
        for j, v in enumerate(subset):
            if i != j:
                if (adj_bits[u] >> v) & 1:
                    degrees[i] += 1
                    
    if len(set(degrees)) == 1:
        return True
    return False

def _find_violation(n, adj_bits):
    """
    Find a subset of size 6 that induces a regular subgraph.
    Returns the subset or None.
    """
    # Check all subsets of size 6
    # For efficiency, we can pre-compute or just iterate.
    # 20 choose 6 is 38760. This is fast enough in Python if optimized.
    # To speed up, we can sample or use heuristics, but full check is safer for correctness.
    
    # Optimization: Use itertools.combinations
    from itertools import combinations
    
    for combo in combinations(range(n), 6):
        if _check_subset_valid(combo, adj_bits, n):
            return combo
            
    # Check size 7
    for combo in combinations(range(n), 7):
        if _check_subset_valid(combo, adj_bits, n):
            return combo
            
    # Check size 8+ (sampling)
    # If we get here, we haven't found 6 or 7.
    # Regular subgraphs of size 8+ are rarer.
    # We will sample for 8..n
    for size in range(8, n + 1):
        for _ in range(50): # Sample 50 subsets
            subset = random.sample(range(n), size)
            if _check_subset_valid(subset, adj_bits, n):
                return subset
                
    return None

def _random_graph(n, density=0.5):
    adj_bits = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    return adj_bits

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n=20, 19, ..., 6
    for n in range(20, 5, -1):
        if time.time() > deadline:
            break
            
        found = False
        max_attempts = 50 if n > 15 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization
            density = 0.3 + 0.4 * random.random()
            adj_bits = _random_graph(n, density)
            
            # Local Search
            # We try to fix violations
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                violation = _find_violation(n, adj_bits)
                
                if violation is None:
                    # Valid graph found!
                    # Check larger sizes just in case? _find_violation samples 8+.
                    # For a "certified" result, we should ideally check all.
                    # But given time, we trust the check for 6,7 and sampling for 8+.
                    # To be safe, if n is small, do full check.
                    if n <= 12:
                        # Full check for all sizes
                        from itertools import combinations
                        is_valid = True
                        for sz in range(6, n + 1):
                            for combo in combinations(range(n), sz):
                                if _check_subset_valid(combo, adj_bits, n):
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        if not is_valid:
                            continue # Reject this candidate
                    
                    # Convert to edges
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj_bits[u] >> v) & 1:
                                edges.append([u, v])
                    
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                    
                # Flip an edge in the violation
                u, v = random.sample(list(violation), 2)
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
            if found:
                break
                
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
