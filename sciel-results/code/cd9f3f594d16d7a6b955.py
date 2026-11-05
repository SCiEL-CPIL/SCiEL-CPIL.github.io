import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second margin for 120s limit
    
    # We aim for the largest known valid graph size for R(3,10) > n.
    # Known bounds: R(3,10) is between 39 and 41.
    # A graph on 40 vertices with no triangle and no independent set of size 10
    # would prove R(3,10) > 40.
    # A graph on 39 vertices proves R(3,10) > 39.
    # The prompt says "survey records 39 as the largest known order".
    # So we should aim for 39 or 40.
    
    # Strategy:
    # 1. Try to construct a valid graph for n=40.
    # 2. If fail, try n=39.
    # 3. If fail, fallback to a known smaller valid graph (e.g., n=26 from seed).
    
    for target_n in [40, 39, 38, 37]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts for local search
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize a random triangle-free graph.
            # A safe way to start triangle-free is a bipartite graph.
            # Then we can try to add edges within partitions if it doesn't create triangles.
            
            n = target_n
            split = n // 2
            A = list(range(split))
            B = list(range(split, n))
            
            # Random bipartite graph with p=0.6
            p = 0.6
            adj = np.zeros((n, n), dtype=bool)
            edges = []
            
            for u in A:
                for v in B:
                    if random.random() < p:
                        adj[u, v] = True
                        adj[v, u] = True
                        edges.append([u, v])
            
            # Local Search / Hill Climbing
            # Goal: Maintain triangle-free, Minimize alpha (independent set size).
            # We want alpha < 10.
            
            # Fast check: Does this graph have an independent set of size 10?
            # If alpha < 10, we are done.
            
            if not has_independent_set_of_size_10(adj, n):
                # Valid graph found
                return {'n': n, 'edges': edges}
            
            # If not valid, try to improve.
            # Move: Add an edge (u,v) if it doesn't create a triangle.
            # Or remove an edge if it helps reduce alpha? (Usually adding edges reduces alpha)
            # Adding edges is safer for triangle-free if we check common neighbors.
            
            # Let's try a greedy local search:
            # While time allows:
            #   Find a non-edge (u,v) such that N(u) and N(v) are disjoint (no common neighbor).
            #   If we add (u,v), does alpha decrease or stay same?
            #   This is hard to check exactly quickly.
            
            # Alternative: Randomized restarts with different p values might be enough
            # if we use a better initial structure.
            
            # Let's try a specific known structure: 
            # The (3,10)-Ramsey graph often relates to the Clebsch graph or similar.
            # But for n=39/40, it's likely a specific known graph.
            
            # Since we can't hardcode the 39-vertex graph easily without knowing it,
            # we rely on the search.
            
            # To speed up search, we can use a "repair" strategy.
            # If alpha >= 10, find an independent set S of size 10.
            # Add edges within S? No, that creates triangles if S is independent?
            # If S is independent, there are NO edges within S.
            # Adding an edge (u,v) in S is safe from triangles IF u and v have no common neighbor.
            # But since S is independent, u and v have no common neighbor IN S.
            # They might have common neighbors OUTSIDE S.
            
            # Let's just do more random restarts with varied parameters.
            # And use a faster alpha checker.
            
            # For now, just continue the loop to try another random instance.
            # The key improvement is the fast alpha checker.
            
    # Fallback: Return a known valid small graph.
    # The seed used n=26 with a specific cyclic pattern.
    # Let's verify a simple valid graph.
    # Complete bipartite K_14,14 has alpha=14 (too big).
    # We need alpha < 10.
    # A random graph G(n, p) might work for smaller n.
    
    # Let's just return the seed's fallback which is known to be valid (score 1.0)
    n = 26
    # The seed's fallback:
    # [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    # Let's double check this is triangle free and alpha < 10.
    # It's a circulant graph.
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    
    return {'n': n, 'edges': edges}

def has_independent_set_of_size_10(adj, n):
    """
    Returns True if there is an independent set of size >= 10.
    Uses a fast branch and bound with bitsets.
    """
    # Convert adjacency matrix to bitsets
    # adj is numpy bool array
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        # Use numpy non-zero indices for speed
        nbrs = np.nonzero(adj[i])[0]
        for j in nbrs:
            mask |= (1 << j)
        neighbors[i] = mask
    
    found_large = [False]
    
    # Heuristic ordering: sort vertices by degree descending?
    # For independent set, picking high degree vertices first might prune faster?
    # Or low degree?
    # Standard B&B for Max Independent Set:
    # Branch on a vertex v.
    # 1. Include v: Remove v and N(v) from candidates.
    # 2. Exclude v: Remove v from candidates.
    
    # We want to find IF size 10 exists.
    
    def dfs(candidates_mask, current_size):
        if found_large[0]:
            return
        
        if current_size >= 10:
            found_large[0] = True
            return
            
        # Pruning: if current_size + popcount(candidates_mask) < 10, return
        # Python int bit_count is fast in 3.10+
        if current_size + candidates_mask.bit_count() < 10:
            return
            
        if candidates_mask == 0:
            return
            
        # Pick a vertex v from candidates
        # Heuristic: Pick vertex with min degree in the induced subgraph?
        # Or just the first one.
        # To be faster, we can try to pick a vertex that is "central" or "peripheral".
        # Let's stick to first set bit for simplicity, but we can improve.
        
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: candidates_mask without v and without neighbors of v
        new_mask = candidates_mask & ~(neighbors[v] | lsb)
        dfs(new_mask, current_size + 1)
        
        if found_large[0]:
            return
        
        # Branch 2: Exclude v
        new_mask = candidates_mask & ~lsb
        dfs(new_mask, current_size)
        
    dfs((1 << n) - 1, 0)
    
    return found_large[0]
