import time
import random
from itertools import combinations

def is_triangle_free(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    return True

def has_independent_set_of_size_k(n, edges, k):
    # Returns True if there is an independent set of size k
    # Uses branch and bound
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Greedy coloring upper bound for independent set?
    # Actually, we want to find an independent set.
    # This is the Maximum Independent Set problem.
    # We can use a recursive search.
    
    best_set = []
    
    def search(candidates, current_set):
        if len(current_set) >= k:
            return True
        if not candidates:
            return False
        if len(current_set) + len(candidates) < k:
            return False
            
        # Heuristic: pick a vertex with high degree in the induced subgraph?
        # Or just pick the first one.
        # To speed up, we can order candidates.
        
        # Branch on the first candidate
        v = candidates[0]
        rest = candidates[1:]
        
        # Include v
        # New candidates are neighbors of v? No, non-neighbors.
        # Independent set: no edges between vertices.
        # If we include v, we cannot include any neighbor of v.
        new_candidates = [u for u in rest if u not in adj[v]]
        if search(new_candidates, current_set + [v]):
            return True
            
        # Exclude v
        if search(rest, current_set):
            return True
            
        return False
        
    return search(list(range(n)), [])

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: Andrasfai graph
    n_and = 26
    edges_and = [[i, j] for i in range(n_and) for j in range(i + 1, n_and) if (j - i) % n_and % 3 == 1]
    
    best_n = n_and
    best_edges = edges_and
    
    # Try to find a larger graph
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 500 # Limit attempts per n
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate a random maximal triangle-free graph
            # Start with empty graph
            adj = [set() for _ in range(n)]
            
            # Add edges randomly
            # To make it dense, we might want to bias towards adding edges.
            # But random order might suffice.
            # We need to ensure it's maximal.
            
            # Simple approach:
            # 1. Randomly permute vertices
            # 2. Add edges if they don't create a triangle
            # This doesn't guarantee maximal.
            
            # Better:
            # Start with a random graph G(n, p) with p=0.5
            # Remove triangles greedily
            # Then try to add edges to make it maximal
            
            edges = []
            adj = [set() for _ in range(n)]
            
            # Random initial edges
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        # Check triangle
                        common = adj[u] & adj[v]
                        if not common:
                            edges.append((u, v))
                            adj[u].add(v)
                            adj[v].add(u)
            
            # Make it maximal: try to add all possible edges
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if v not in adj[u]:
                            common = adj[u] & adj[v]
                            if not common:
                                edges.append((u, v))
                                adj[u].add(v)
                                adj[v].add(u)
                                changed = True
            
            # Check if valid
            # 1. Triangle-free (guaranteed by construction)
            # 2. Alpha < 10
            
            # Check Alpha
            # We need to check if there is an independent set of size 10
            # has_independent_set_of_size_k(n, edges, 10)
            
            # Optimization: If we find an independent set of size 10, it's invalid.
            # If we don't, it might be valid.
            # But the B&B might be slow.
            
            # Let's try a greedy independent set first to see if we can easily find size 10.
            # If we can't find 10 greedily, it's a good candidate.
            
            # Greedy Independent Set
            remaining = set(range(n))
            indep = []
            while remaining:
                # Pick vertex with min degree in remaining?
                # Or just random
                v = random.choice(list(remaining))
                indep.append(v)
                remaining.discard(v)
                remaining -= adj[v]
                
            if len(indep) < 10:
                # Greedy found < 10. Likely valid.
                # Verify with B&B
                if not has_independent_set_of_size_k(n, edges, 10):
                    # Valid!
                    best_n = n
                    best_edges = [[u, v] for u, v in edges]
                    found = True
                    break
            
            # If greedy found >= 10, try to break it?
            # Or just restart.
            
        if found:
            break
            
    return {'n': best_n, 'edges': [[u, v] for u, v in best_edges]}
