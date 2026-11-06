import time
import math
import random

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # 1. Identify valid vertices
    # Constraint: 2*a is NOT a perfect square
    valid_vertices = []
    for i in range(1, 257):
        if not is_square(2 * i):
            valid_vertices.append(i)
            
    N = len(valid_vertices)
    # Map value to index for faster lookup? Or just use values directly in sets.
    # Using values directly is fine for N=256.
    
    # 2. Build adjacency
    # adj[u] = list of v in valid_vertices such that u+v is square
    squares = [k*k for k in range(1, 23)] # 1 to 529
    
    adj = {v: [] for v in valid_vertices}
    for u in valid_vertices:
        for sq in squares:
            v = sq - u
            if v in adj and v != u: # v must be valid and distinct
                adj[u].append(v)
                
    # Ensure symmetry is handled correctly in logic, 
    # but for conflict counting, we just need to know neighbors.
    # adj[u] contains neighbors.
    
    best_set = set()
    best_size = 0
    
    # Helper: Greedy Min-Degree MIS
    def greedy_mis(vertices, adj):
        avail = set(vertices)
        S = set()
        while avail:
            # Find vertex with min degree in induced subgraph of avail
            min_deg = float('inf')
            min_v = None
            for u in avail:
                deg = 0
                for w in adj[u]:
                    if w in avail:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = u
            
            S.add(min_v)
            # Remove min_v and its neighbors from avail
            to_remove = set(adj[min_v])
            to_remove.add(min_v)
            avail -= to_remove
            
        return S

    # Helper: Local Search Improvement
    def local_search(S, adj, valid_vertices, deadline):
        S = set(S)
        current_size = len(S)
        
        # Precompute list of vertices not in S for iteration
        # We will recompute conflicts each step for simplicity and correctness
        
        while time.time() < deadline:
            # Compute conflict counts for v not in S
            conflicts = {v: 0 for v in valid_vertices if v not in S}
            
            for u in S:
                for w in adj[u]:
                    if w not in S:
                        conflicts[w] += 1
            
            # 1. Try to expand
            expanded = False
            for v, c in conflicts.items():
                if c == 0:
                    S.add(v)
                    current_size += 1
                    expanded = True
                    break
            
            if expanded:
                continue
            
            # 2. Try to swap (1-swap)
            # Find v not in S with conflicts[v] == 1
            # The unique neighbor u in S must be removed
            swap_candidates = []
            for v, c in conflicts.items():
                if c == 1:
                    # Find the neighbor u in S
                    for u in adj[v]:
                        if u in S:
                            swap_candidates.append((u, v))
                            break
                            
            if swap_candidates:
                # Pick a random candidate to avoid cycles
                u, v = random.choice(swap_candidates)
                S.remove(u)
                S.add(v)
                # Size remains same, but structure changes
                # We don't update current_size
                continue
            
            # 3. Stuck in this local optimum
            # Try a 2-swap or random perturbation?
            # For now, break to try a new random restart or greedy init
            break
            
        return S, current_size

    # Main Loop
    # Run multiple random restarts of Greedy + Local Search
    # Greedy is deterministic, so we need randomness in Greedy or multiple inits.
    # Let's add randomness to Greedy: among min-degree vertices, pick one randomly.
    
    def greedy_mis_random(vertices, adj):
        avail = set(vertices)
        S = set()
        while avail:
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = 0
                for w in adj[u]:
                    if w in avail:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            min_v = random.choice(candidates)
            S.add(min_v)
            to_remove = set(adj[min_v])
            to_remove.add(min_v)
            avail -= to_remove
            
        return S

    # Also try simple random inits? No, greedy is better.
    
    while time.time() < deadline:
        # Get initial set
        S_init = greedy_mis_random(valid_vertices, adj)
        
        # Improve
        S_improved, size = local_search(S_init, adj, valid_vertices, deadline)
        
        if size > best_size:
            best_size = size
            best_set = S_improved
            
        # If we found a very large set, maybe we can stop early?
        # Max possible is 256. 
        # Heuristic bound? 
        # Just run until deadline.

    if not best_set:
        # Fallback
        best_set = {1}
        
    return list(best_set)
