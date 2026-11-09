import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 17
    best_edges = None
    
    # Seed graph: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17
    best_n = 17
    
    # Try to find better graphs for n > 17
    # We search for circulant graphs
    # n from 35 down to 18
    
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
            
        # Local search for circulant graph on n vertices
        max_dist = n // 2
        # State is a bitmask of length max_dist
        # dist d is present if bit d is set (0-indexed, so bit 0 is dist 1? No, let's use 1-indexed dist)
        # Let's use a set of distances or a bitmask.
        # Bitmask: bit i represents distance i+1.
        
        # Initialize state
        # Try a few random inits
        found = False
        for init_iter in range(50):
            if time.time() > deadline:
                break
                
            if init_iter == 0:
                # Try a "Paley-like" init if n is prime?
                # Just random for now
                state = 0
                for d in range(1, max_dist + 1):
                    if random.random() < 0.5:
                        state |= (1 << d)
            else:
                state = random.getrandbits(max_dist + 1)
                
            # Local search
            current_state = state
            # Calculate initial cost
            adj = build_adj(current_state, n)
            cost = calculate_cost(adj, n)
            
            if cost == 0:
                found = True
                best_n = n
                best_edges = get_edges_from_adj(adj, n)
                break
            
            # Simulated Annealing / Hill Climbing
            temp = 1.0
            for step in range(10000):
                if time.time() > deadline:
                    break
                    
                # Pick a random distance to flip
                d = random.randint(1, max_dist)
                new_state = current_state ^ (1 << d)
                new_adj = build_adj(new_state, n)
                new_cost = calculate_cost(new_adj, n)
                
                if new_cost < cost:
                    current_state = new_state
                    adj = new_adj
                    cost = new_cost
                    if cost == 0:
                        found = True
                        best_n = n
                        best_edges = get_edges_from_adj(adj, n)
                        break
                elif new_cost == cost:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.5:
                        current_state = new_state
                        adj = new_adj
                else:
                    # Accept with probability exp(-delta/temp)
                    delta = new_cost - cost
                    if random.random() < pow(2.718, -delta / temp):
                        current_state = new_state
                        adj = new_adj
                        cost = new_cost
                temp *= 0.999
                if temp < 0.01:
                    break
            
            if found:
                break
        
        if found:
            break
    
    return {'n': best_n, 'edges': best_edges}

def build_adj(state, n):
    adj = [0] * n
    max_dist = n // 2
    for d in range(1, max_dist + 1):
        if state & (1 << d):
            for u in range(n):
                v1 = (u + d) % n
                v2 = (u - d) % n
                adj[u] |= (1 << v1)
                adj[u] |= (1 << v2)
    return adj

def get_edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def calculate_cost(adj, n):
    # Check K4
    if has_k4(adj, n):
        return 1
    # Check I6 (K6 in complement)
    if has_k6_complement(adj, n):
        return 1
    return 0

def has_k4(adj, n):
    # For each pair (u,v) with edge, check common neighbors
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                # Remove u and v
                common &= ~(1 << u)
                common &= ~(1 << v)
                # If common has at least 2 bits, check if any two are connected
                # Get list of common neighbors
                neighbors = []
                temp = common
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    neighbors.append(k)
                    temp ^= lsb
                if len(neighbors) >= 2:
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            k1 = neighbors[i]
                            k2 = neighbors[j]
                            if adj[k1] & (1 << k2):
                                return True
    return False

def has_k6_complement(adj, n):
    # Build complement adj
    mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = (mask ^ adj[i]) & ~(1 << i)
    
    # Check for K6 in comp_adj
    # This is expensive. 
    # Optimization: If alpha(adj) < 6, return False.
    # We can use a simple greedy bound? No, must be exact.
    # But we can prune.
    # Let's use a backtracking search for clique of size 6.
    
    # To speed up, we can order vertices by degree in comp_adj?
    # Or just check all 6-subsets? 
    # n=35, 1.6M subsets. 
    # In Python, 1.6M iterations is ~0.1-0.5s. 
    # If we do this 10000 times in local search, it's 1000-5000s. Too slow.
    
    # We need a faster check or fewer checks.
    # Maybe check K4 first (fast). If K4 exists, cost=1, skip K6.
    # If no K4, then check K6.
    # But many states will have K4.
    # What fraction of random graphs have no K4?
    # For n=35, p=0.5, expected K4 is huge.
    # So most states will be rejected by K4 check quickly.
    # Only states with low density might pass K4.
    # If we start with random p=0.5, we will mostly hit K4.
    # We should start with lower density?
    # Or just accept that K6 check is only run on sparse graphs.
    
    # Let's optimize K6 check.
    # We can use a recursive function to find a clique of size 6.
    
    def has_clique_size_k(adj, k):
        # adj is list of bitmasks
        # vertices 0..n-1
        # Use backtracking
        # candidates: list of vertices
        # current_clique: list of vertices
        
        # To make it fast, we can use bitsets.
        # candidates is a bitmask.
        
        def backtrack(candidates, current_size):
            if current_size == k:
                return True
            if candidates == 0:
                return False
            # Pruning: if number of candidates < k - current_size, return False
            if candidates.bit_count() < k - current_size:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with highest degree in induced subgraph?
            # Or just pick first.
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates_without_v = candidates ^ lsb
            
            # Option 1: Include v
            # New candidates are neighbors of v in candidates_without_v
            new_candidates = candidates_without_v & adj[v]
            if backtrack(new_candidates, current_size + 1):
                return True
            
            # Option 2: Exclude v
            if backtrack(candidates_without_v, current_size):
                return True
                
            return False
        
        initial_candidates = (1 << n) - 1
        return backtrack(initial_candidates, 0)
        
    return has_clique_size_k(comp_adj, 6)
