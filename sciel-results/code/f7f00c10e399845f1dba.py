import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # 1. Find all bad triples and build the conflict graph G
    # G[u] is a bitmask of v such that there exists w where {u,v,w} is a bad triple.
    # Note: The existence of w is what makes u,v "conflicting" in the graph sense.
    conflict_graph = [0] * (N + 1)
    bad_triple_masks = []
    
    # Optimization: Precompute lcm table? N=128, 128*128 is small.
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            lcm_table[a][b] = lcm(a, b)
            
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lab == lcm_table[a][c] == lcm_table[b][c]:
                    mask = (1 << a) | (1 << b) | (1 << c)
                    bad_triple_masks.append(mask)
                    # Update conflict graph
                    conflict_graph[a] |= (1 << b)
                    conflict_graph[b] |= (1 << a)
                    conflict_graph[a] |= (1 << c)
                    conflict_graph[c] |= (1 << a)
                    conflict_graph[b] |= (1 << c)
                    conflict_graph[c] |= (1 << b)
                    
    if not bad_triple_masks:
        return [i for i in range(1, N + 1) if full_mask & (1 << i)]

    def is_valid(mask):
        for t in bad_triple_masks:
            if (mask & t) == t:
                return False
        return True
        
    def get_degree(mask, u):
        # Degree of u in the induced subgraph defined by mask
        # Count bits in conflict_graph[u] & mask
        return (conflict_graph[u] & mask).bit_count()

    def greedy_mis_on_graph():
        # Greedy Independent Set on the conflict graph G.
        # Strategy: Always pick the vertex with the minimum degree in the remaining graph.
        # This tends to preserve more vertices.
        remaining = full_mask
        selected = 0
        
        while remaining:
            # Find vertex with min degree in remaining
            min_deg = N + 1
            best_u = 0
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = (conflict_graph[u] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_u = u
                temp ^= lsb
                
            if best_u == 0:
                break
                
            selected |= (1 << best_u)
            # Remove best_u and all its neighbors from remaining
            remaining &= ~(1 << best_u)
            remaining &= ~conflict_graph[best_u]
            
        return selected

    def local_search_graph(mask):
        # Improve the independent set in G using 1-swap (remove one, add one) 
        # to potentially allow adding more later? 
        # Actually, standard MIS local search is complex. 
        # We will rely on the hypergraph-aware local search from the original program.
        # But we can do a simple "shake" to escape local minima in G?
        # Let's stick to the hypergraph polish which is more powerful.
        return mask

    def add_back(mask):
        improved = True
        while improved:
            improved = False
            for i in range(1, N + 1):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    if is_valid(new_mask):
                        mask = new_mask
                        improved = True
        return mask
        
    def swap_improve(mask):
        # 1 out, 2 in
        improved = True
        while improved:
            improved = False
            in_mask = [i for i in range(1, N + 1) if mask & (1 << i)]
            not_in_mask = [i for i in range(1, N + 1) if not (mask & (1 << i))]
            
            if not not_in_mask:
                break
                
            # Sort not_in_mask to make it deterministic?
            # Try all pairs
            for i in range(len(not_in_mask)):
                v = not_in_mask[i]
                for j in range(i + 1, len(not_in_mask)):
                    w = not_in_mask[j]
                    
                    M_prime = mask | (1 << v) | (1 << w)
                    # Find all bad triples contained in M_prime
                    blocking_triples = [t for t in bad_triple_masks if (M_prime & t) == t]
                    
                    if not blocking_triples:
                        if is_valid(M_prime):
                            mask = M_prime
                            improved = True
                            break
                        # If M_prime is invalid, we MUST remove an element.
                    
                    # We need to find a u in in_mask such that removing u fixes all blocking_triples.
                    # This means u must be in ALL blocking_triples.
                    # Intersection of all blocking_triples.
                    common_intersection = None
                    for t in blocking_triples:
                        if common_intersection is None:
                            common_intersection = t
                        else:
                            common_intersection &= t
                    
                    if common_intersection is None:
                        continue
                        
                    # u must be in common_intersection AND in in_mask
                    candidate_u = None
                    x = common_intersection
                    while x:
                        lsb = x & -x
                        idx = lsb.bit_length() - 1
                        if mask & (1 << idx):
                            candidate_u = idx
                            break
                        x ^= lsb
                    
                    if candidate_u is not None:
                        mask = (mask & ~(1 << candidate_u)) | (1 << v) | (1 << w)
                        improved = True
                        break
            
            if improved:
                continue
                
        return mask

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy MIS on the conflict graph
    m = greedy_mis_on_graph()
    m = add_back(m)
    m = swap_improve(m)
    size = m.bit_count()
    if size > best_size:
        best_size = size
        best_mask = m
        
    # Strategy 2: Randomized Greedy MIS on the conflict graph
    random.seed(42)
    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        
        # Randomized greedy: pick min degree, but with some randomness among ties
        remaining = full_mask
        selected = 0
        
        while remaining:
            # Find min degree
            min_deg = N + 1
            candidates = []
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = (conflict_graph[u] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
                temp ^= lsb
                
            if not candidates:
                break
                
            u = random.choice(candidates)
            selected |= (1 << u)
            remaining &= ~(1 << u)
            remaining &= ~conflict_graph[u]
            
        m = selected
        m = add_back(m)
        m = swap_improve(m)
        size = m.bit_count()
        if size > best_size:
            best_size = size
            best_mask = m
            
    # Strategy 3: Original Greedy Removal (from current program) as a fallback/different perspective
    def greedy_remove(full_mask, randomize=False):
        mask = full_mask
        while True:
            active_triples = [t for t in bad_triple_masks if (mask & t) == t]
            if not active_triples:
                break
            
            counts = [0] * (N + 1)
            for t in active_triples:
                x = t
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    x ^= lsb
            
            max_count = 0
            candidates = []
            for i in range(1, N + 1):
                if mask & (1 << i):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        candidates = [i]
                    elif counts[i] == max_count:
                        candidates.append(i)
            
            if not candidates:
                break
                
            if randomize:
                u = random.choice(candidates)
            else:
                u = candidates[0]
            
            mask &= ~(1 << u)
            
            if time.time() - start_time > deadline - 5:
                break
        
        return mask

    # Run a few original greedy removals
    for _ in range(20):
        if time.time() - start_time > deadline - 5:
            break
        m = greedy_remove(full_mask, randomize=True)
        m = add_back(m)
        m = swap_improve(m)
        size = m.bit_count()
        if size > best_size:
            best_size = size
            best_mask = m

    # Convert mask to list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    return result
