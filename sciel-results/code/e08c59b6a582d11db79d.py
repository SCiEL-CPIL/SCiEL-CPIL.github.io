import random
import time

def construct():
    d = 7
    n = 1 << d
    # Precompute all edges with their dimension
    edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v, i))
    
    # Helper to check if adding edge (u,v) of dimension i creates a C4
    def is_valid(chosen, u, v, i):
        for j in range(d):
            if j == i:
                continue
            uj = u ^ (1 << j)
            vj = v ^ (1 << j)
            # The other three edges of the face spanned by dims i and j
            if (min(u, uj), max(u, uj)) in chosen and \
               (min(v, vj), max(v, vj)) in chosen and \
               (min(uj, vj), max(uj, vj)) in chosen:
                return False
        return True

    best_size = 0
    best_edges = []
    
    # Use a fixed seed for reproducibility in case the evaluator runs multiple times?
    # Or just let it be random. The task says "search score is value/seed", implying the seed program is the baseline.
    # Randomness is usually fine.
    
    start_time = time.time()
    max_time = 110  # Leave buffer for overhead
    
    # Run multiple randomized greedy passes
    for _ in range(10000):
        if time.time() - start_time > max_time:
            break
        
        chosen = set()
        current_edges = []
        
        # Create a random order of edges
        # To avoid creating a new list each time, we can shuffle a copy
        # But edges is a list of tuples.
        # random.shuffle modifies in place, so we need a copy.
        # 448 edges is small.
        shuffled_edges = edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v, i in shuffled_edges:
            if is_valid(chosen, u, v, i):
                chosen.add((min(u, v), max(u, v)))
                current_edges.append([u, v])
        
        if len(current_edges) > best_size:
            best_size = len(current_edges)
            best_edges = current_edges
            
    return best_edges
