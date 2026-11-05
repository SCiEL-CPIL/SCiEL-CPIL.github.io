import time
import random
import math

def _has_triangle(n, adj_mask):
    """Check if graph with adjacency masks has a triangle."""
    for i in range(n):
        # Check edges (i, j) where j > i
        # And (i, k) where k > j
        # If (j, k) is edge, triangle exists.
        mask_i = adj_mask[i]
        # Iterate j in neighbors of i
        tmp = mask_i
        while tmp:
            lsb_j = tmp & (-tmp)
            j = lsb_j.bit_length() - 1
            if j > i:
                # Check if any neighbor of j is also neighbor of i and > j?
                # Simpler: intersection of neighbors of i and j
                # If (adj_mask[i] & adj_mask[j]) != 0, there is a common neighbor.
                # Since graph is simple, if common neighbor exists, triangle exists.
                if adj_mask[i] & adj_mask[j]:
                    return True
            tmp ^= lsb_j
    return False

def _alpha_bitmask(n, adj_mask):
    """Calculate independence number using branch and bound on bitmasks."""
    full_mask = (1 << n) - 1
    # comp_mask[i] = vertices not adjacent to i and not i
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = [0]
    
    def dfs(candidates, current_size):
        if current_size + candidates.bit_count() <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Heuristic: pick vertex with max degree in candidate subgraph?
        # Or min degree?
        # For independent set, we want to include vertices that don't exclude many others.
        # So pick vertex with min (candidates & comp_mask[v]).bit_count()?
        # No, that's the number of *available* neighbors?
        # We want to keep options open.
        # Standard: pick vertex with max degree in the *induced subgraph* of candidates?
        # Actually, for independent set, picking a vertex removes its neighbors.
        # We want to remove few neighbors.
        # So pick v that minimizes (candidates & adj_mask[v]).bit_count()?
        # i.e., v has fewest neighbors in candidates.
        
        cands_list = []
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cands_list.append(v)
            tmp ^= lsb
            
        # Sort by number of neighbors in candidates (ascending)
        # v with fewest neighbors is "safer" to include?
        # If we include v, we remove its neighbors.
        # If v has few neighbors, we remove few candidates.
        cands_list.sort(key=lambda v: (candidates & adj_mask[v]).bit_count())
        
        v = cands_list[0]
        
        # Include v
        dfs(candidates & comp_mask[v] & ~(1 << v), current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
        
    dfs(full_mask, 0)
    return best[0]

def _build_adj_mask(n, S):
    adj = [0] * n
    S_set = set(S)
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    return adj

def _is_valid(n, S):
    adj = _build_adj_mask(n, S)
    if _has_triangle(n, adj):
        return False
    alpha = _alpha_bitmask(n, adj)
    return alpha < 10

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Try n=40 first
    for n in [40, 39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break
            
        found = False
        
        # Strategy 1: Random search
        for _ in range(200):
            if time.time() > deadline:
                break
            random.seed()
            # Generate random symmetric set
            # Density around 0.2-0.3
            density = random.uniform(0.15, 0.35)
            target_size = int(n * density)
            half = n // 2
            available = list(range(1, half + 1))
            random.shuffle(available)
            
            S = set()
            for x in available:
                if len(S) >= target_size:
                    break
                neg_x = (-x) % n
                new_S = S | {x, neg_x}
                # Quick triangle check on new_S?
                # Just build and check
                if _is_valid(n, new_S):
                    # Found valid graph
                    edges = []
                    S_set = set(new_S)
                    for i in range(n):
                        for s in new_S:
                            j = (i + s) % n
                            if j > i:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            # If we didn't find a valid one in this random set, we can try to repair it?
            # For now, just try next random set.
            
        # Strategy 2: Structured sets
        # Try quadratic residues if n is prime
        if n > 2:
            is_prime = True
            for p in range(2, int(math.sqrt(n)) + 1):
                if n % p == 0:
                    is_prime = False
                    break
            if is_prime:
                S = set()
                for x in range(1, n):
                    if (x * x) % n < n // 2:
                        S.add(x)
                        S.add((-x) % n)
                if _is_valid(n, S):
                    edges = []
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if j > i:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                    
        # Strategy 3: Local Search on a random graph
        # Start with a random triangle-free graph
        # Try to add edges to reduce alpha
        for _ in range(10):
            if time.time() > deadline:
                break
            # Generate random triangle-free graph
            # Use greedy: add edge if no triangle formed
            adj = [0] * n
            edges = []
            random.seed(n + _)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.2:
                        # Check if adding (i,j) creates triangle
                        # Common neighbors of i and j?
                        if not (adj[i] & adj[j]):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            edges.append([i, j])
            
            # Check if valid
            if _has_triangle(n, adj):
                continue # Should not happen if greedy was correct, but safe
                
            alpha = _alpha_bitmask(n, adj)
            if alpha < 10:
                return {'n': n, 'edges': edges}
            
            # Local search: try to add edges
            # This is complex to implement correctly in short time.
            # Skip for now.

    # Fallback: Return a small valid graph
    # n=16 Clebsch
    n = 16
    edges = []
    # Clebsch graph construction
    # Vertices 0..15
    # Edges if Hamming distance of binary representation is 1 or 3?
    # Clebsch is complement of Shrikhande? No.
    # Clebsch: vertices are binary strings of length 4.
    # Edge if Hamming distance is 1 or 3.
    for i in range(16):
        for j in range(i + 1, 16):
            xor = i ^ j
            bits = bin(xor).count('1')
            if bits == 1 or bits == 3:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
