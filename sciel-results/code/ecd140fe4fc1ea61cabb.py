import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    rng = random.Random(42)
    
    best_result = None
    best_n = 0
    
    # Try n=22, 21, 20
    for n in [22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        # Allocate time
        remaining = deadline - time.time()
        target_time = min(30.0, remaining - 2.0)
        if target_time <= 0:
            break
            
        result = solve_n(n, target_time, rng)
        
        if result is not None:
            # Found a valid graph for this n
            # Since we go from high to low, the first one found is the best
            return result
            
    # If we didn't find any, return a small valid graph
    # n=5 is always valid
    if best_result is None:
        return {'n': 5, 'edges': []}
        
    return best_result

def solve_n(n, time_limit, rng):
    """
    Use Simulated Annealing to find a valid graph on n vertices.
    """
    start_time = time.time()
    
    # Precompute all 6-subsets
    all_subsets = list(combinations(range(n), 6))
    
    # Precompute for each pair (u, v), the list of subsets containing u and v
    # subset_index: index in all_subsets
    # pair_to_subsets: dict mapping (u, v) to list of subset indices
    pair_to_subsets = {}
    for idx, subset in enumerate(all_subsets):
        # Generate all pairs in this subset
        for i in range(6):
            for j in range(i+1, 6):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                key = (u, v)
                if key not in pair_to_subsets:
                    pair_to_subsets[key] = []
                pair_to_subsets[key].append(idx)
    
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Calculate initial energy
    energy = count_violations(adj, n, all_subsets)
    
    if energy == 0:
        return to_result(adj, n)
        
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    current_energy = energy
    
    while time.time() < start_time + time_limit:
        if T < min_T:
            # Restart with new random graph if stuck too long?
            # Or just break?
            # Let's try a few restarts
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            current_energy = count_violations(adj, n, all_subsets)
            if current_energy == 0:
                return to_result(adj, n)
            T = 1.0
            cooling = 0.999
            
        # Pick a random edge to flip
        u, v = rng.sample(range(n), 2)
        
        # Calculate delta energy
        key = (u, v) if u < v else (v, u)
        affected_subsets = pair_to_subsets.get(key, [])
        
        delta_e = 0
        # Check each affected subset
        for idx in affected_subsets:
            subset = all_subsets[idx]
            # Create mask for subset
            mask = 0
            for x in subset:
                mask |= (1 << x)
                
            # Check if regular BEFORE flip
            # Note: adj currently has the OLD state
            is_reg_before = is_regular_subset(adj, mask)
            
            # Temporarily flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            # Check if regular AFTER flip
            is_reg_after = is_regular_subset(adj, mask)
            
            # Revert flip
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            if is_reg_before and not is_reg_after:
                delta_e -= 1
            elif not is_reg_before and is_reg_after:
                delta_e += 1
            # else: no change in count for this subset
            
        # Acceptance criterion
        if delta_e < 0:
            # Better
            pass
        else:
            # Worse or same
            import math
            if T > 0:
                prob = math.exp(-delta_e / T)
                if rng.random() > prob:
                    # Reject
                    T *= cooling
                    continue
            
        # Apply flip
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        current_energy += delta_e
        
        if current_energy == 0:
            return to_result(adj, n)
            
        T *= cooling
        
    return None

def is_regular_subset(adj, mask):
    """Check if induced subgraph on mask is regular."""
    # Get vertices in mask
    # We need degrees of all vertices in mask within mask
    # We can iterate bits
    degs = []
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        deg = (adj[v] & mask).bit_count()
        degs.append(deg)
        m ^= lsb
        
    if not degs:
        return True
    d0 = degs[0]
    for d in degs[1:]:
        if d != d0:
            return False
    return True

def count_violations(adj, n, all_subsets):
    count = 0
    for subset in all_subsets:
        mask = 0
        for x in subset:
            mask |= (1 << x)
        if is_regular_subset(adj, mask):
            count += 1
    return count

def to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
