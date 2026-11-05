import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 128
    # 1. Precompute bad triples
    bad_triples = []
    # Optimization: use list of tuples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) and lab == lcm(b, c):
                    bad_triples.append((a, b, c))
    
    # 2. Initialize degrees and adjacency
    # degree[x] = number of bad triples containing x that are "active" (all nodes in S)
    # Actually, simpler: just count total bad triples containing x initially.
    # But we need to update.
    
    # Let's use a set for S
    S = set(range(1, N + 1))
    
    # Pre-compute for each node, the list of triples it belongs to
    node_triples = {i: [] for i in range(1, N + 1)}
    for t in bad_triples:
        for x in t:
            node_triples[x].append(t)
    
    # Initial degrees
    degree = {i: len(node_triples[i]) for i in range(1, N + 1)}
    
    # Helper to check if a triple is active
    def is_active(triple):
        return all(x in S for x in triple)
    
    # Main loop
    while time.time() < deadline:
        # Find an active bad triple
        active_triple = None
        for t in bad_triples:
            if is_active(t):
                active_triple = t
                break
        
        if active_triple is None:
            break
            
        a, b, c = active_triple
        
        # Calculate current degrees for a, b, c
        # Note: degree[x] might be stale if we haven't updated it properly.
        # Let's just re-calculate the "active degree" for a, b, c on the fly?
        # Or maintain it properly.
        
        # Proper maintenance:
        # When we remove a node, we update degrees of its neighbors in the hypergraph.
        
        # Let's just pick the one with the highest *initial* degree?
        # Or re-calculate active degree for a,b,c.
        # Active degree of x = count of triples in node_triples[x] that are active.
        
        def get_active_degree(x):
            count = 0
            for t in node_triples[x]:
                if is_active(t):
                    count += 1
            return count
            
        da = get_active_degree(a)
        db = get_active_degree(b)
        dc = get_active_degree(c)
        
        # Remove the one with highest active degree
        # Tie-breaker: remove the one with highest value? Or lowest?
        # Let's try highest value to keep small numbers (primes) which might be more constraining?
        # Actually, small numbers appear in more LCMs?
        # Let's just remove the max degree.
        
        candidates = [(da, a), (db, b), (dc, c)]
        # Sort by degree desc, then value desc
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        
        victim = candidates[0][1]
        S.remove(victim)
        
        # Update degrees?
        # If we don't update degrees, get_active_degree will be slow but correct.
        # Given N=128 and likely < 1000 bad triples, this is fine.
        
    return list(S)
