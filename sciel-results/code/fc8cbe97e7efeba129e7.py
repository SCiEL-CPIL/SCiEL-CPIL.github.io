import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Precompute subsets for n=10..20 to speed up search
    # Note: subsets depend on n, so we compute on the fly or cache
    subset_cache = {}
    def get_subsets(n, k):
        key = (n, k)
        if key not in subset_cache:
            subset_cache[key] = list(combinations(range(n), k))
        return subset_cache[key]

    # Try fixed-seed random graphs for n=20 down to n=10
    for target_n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break

        # Try multiple restarts for this n
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            result = _local_search(target_n, deadline - 2, get_subsets)
            if result is not None:
                return result

    # Fallback: return a trivially valid small graph
    return {'n': 5, 'edges': []}

def _local_search(n, deadline, get_subsets):
    rng = random.Random()
    
    # Popcount helper
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and rng.random() < 0.5:
                mask |= (1 << j)
        adj[i] = mask

    def flip_edge(u, v):
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def is_regular(subset):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        first_deg = -1
        for v in subset:
            d = popcount(adj[v] & mask)
            if first_deg == -1:
                first_deg = d
            elif d != first_deg:
                return False
        return True

    def find_violation(start_k=6):
        for k in range(start_k, n + 1):
            subs = get_subsets(n, k)
            for subset in subs:
                if is_regular(subset):
                    return subset
        return None

    for iteration in range(2000):
        if time.time() > deadline:
            break

        # Check size 6 first (most common)
        viol = find_violation(6)
        
        if viol is not None:
            # If it's size 6, we know it's 6. If larger, it's larger.
            # We just flip a random edge in it.
            u, v = rng.sample(list(viol), 2)
            flip_edge(u, v)
        else:
            # No violation found?
            # This implies the graph is valid.
            return _to_result(adj, n)

        # Escape local minima
        if iteration % 50 == 0 and rng.random() < 0.05:
            u, v = rng.sample(range(n), 2)
            flip_edge(u, v)

    # Final check
    if find_violation(6) is None:
        return _to_result(adj, n)
        
    return None

def _to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
