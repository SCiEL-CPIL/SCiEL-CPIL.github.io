import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    # Helper to check if a specific mask induces a regular subgraph
    def is_regular(mask, adj):
        k = mask.bit_count()
        if k < 6:
            return False
        
        # Calculate degrees for vertices in mask
        # We can stop early if we find non-regularity
        # But we need to check all vertices to be sure it's regular
        # Optimization: if k is small, just compute all.
        
        # Get list of vertices in mask
        # Using a loop to extract bits is fast in Python 3.11+ but let's stick to compatible code
        # Actually, for small k (6-22), extracting bits is fine.
        
        degrees = []
        m = mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Count neighbors in mask
            # adj[v] is bitmask of neighbors of v
            # Intersection with mask gives neighbors within the subset
            # Popcount gives degree
            deg = (adj[v] & mask).bit_count()
            degrees.append(deg)
            
            # Early exit if we already see two different degrees?
            # We can't just compare to the first one if we haven't seen all.
            # But we can track min and max.
            # Or just store in list and check at end.
            # For k<=22, list is small.
            
        if not degrees:
            return False
        
        d0 = degrees[0]
        for d in degrees[1:]:
            if d != d0:
                return False
        return True

    # Count how many bad subsets exist
    def count_bad(n, adj):
        bad_count = 0
        # Iterate all masks
        # We can skip masks with popcount < 6
        # Generating combinations is faster than iterating all 2^n masks and filtering
        # For n=20, 2^20 = 1M. Filtering is cheap.
        # But itertools.combinations might be faster for specific sizes?
        # Let's stick to mask iteration for simplicity, or use combinations.
        
        # Using combinations is better because we don't iterate over 2^n if n is large,
        # but for n=21, 2^21 is 2M, which is fine.
        # However, generating combinations explicitly avoids the popcount check for every mask.
        
        # Let's use combinations for efficiency in Python
        from itertools import combinations
        
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                # Build mask
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                if is_regular(mask, adj):
                    bad_count += 1
                    # We could break early if we just want to know if it's valid (0 bad)
                    # But for SA, we need the count to guide the search.
                    # If bad_count becomes large, we can break? No, we need the gradient.
                    # But if bad_count > 100, it's clearly bad.
        
        return bad_count

    # Faster count for "is valid" (returns bool)
    def is_valid_graph(n, adj):
        from itertools import combinations
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular(mask, adj):
                    return False
        return True

    def search(n, time_limit):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_bad = count_bad(n, adj)
        best_adj = adj[:]
        best_bad = current_bad
        
        T = 1.0
        cooling = 0.999
        
        while time.time() < time_limit:
            if current_bad == 0:
                return adj
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_bad = count_bad(n, adj)
            
            if new_bad < current_bad:
                current_bad = new_bad
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_adj = adj[:]
            else:
                # Accept with probability
                if new_bad > current_bad:
                    # Simple acceptance: if close, accept
                    prob = 1.0 / (1.0 + (new_bad - current_bad))
                    if random.random() < prob:
                        current_bad = new_bad
                    else:
                        # Revert
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                else: # Equal
                    current_bad = new_bad
            
            T *= cooling
            
        if best_bad == 0:
            return best_adj
        return None

    # Try n=21
    t_start = time.time()
    time_for_21 = min(10, deadline - t_start)
    if time_for_21 > 1:
        res = search(21, t_start + time_for_21)
        if res:
            edges = []
            for i in range(21):
                for j in range(i + 1, 21):
                    if res[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 21, 'edges': edges}

    # Try n=20
    t_start = time.time()
    time_for_20 = min(30, deadline - t_start)
    if time_for_20 > 1:
        res = search(20, t_start + time_for_20)
        if res:
            edges = []
            for i in range(20):
                for j in range(i + 1, 20):
                    if res[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 20, 'edges': edges}

    # Try n=19
    t_start = time.time()
    time_for_19 = min(20, deadline - t_start)
    if time_for_19 > 1:
        res = search(19, t_start + time_for_19)
        if res:
            edges = []
            for i in range(19):
                for j in range(i + 1, 19):
                    if res[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 19, 'edges': edges}

    # Fallback: Small valid graph
    # Empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}
