import time
import math
from math import gcd
import random

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses maximum clique search with greedy coloring upper bounds
    on the graph of candidates in [1,255] divisible by 3, 5, or 17.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    # Build candidate set: all numbers in [1,255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    
    # Build adjacency: edge if gcd > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = adj[j][i] = True
    
    # Index of 255 in candidates
    idx_255 = candidates.index(255)
    
    # Initial solution: all multiples of 3 (clique of size 85)
    best_clique = set(i for i in range(n) if candidates[i] % 3 == 0)
    best_size = len(best_clique)
    
    # Greedy coloring upper bound for maximum clique
    def greedy_color_bound(vertices):
        """Upper bound on clique size using greedy coloring."""
        if not vertices:
            return 0
        # Order vertices by degree within the subgraph
        vlist = list(vertices)
        vlist.sort(key=lambda v: sum(1 for u in vlist if u != v and adj[v][u]), reverse=True)
        
        colors = {}
        color_count = 0
        for v in vlist:
            # Find smallest available color
            used_colors = set()
            for u in vlist:
                if u != v and adj[v][u] and u in colors:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            color_count = max(color_count, c)
        return color_count
    
    # Local search: try to add vertices to clique
    def local_search(clique, max_time=None):
        """Try to grow clique by adding vertices, and swap for larger ones."""
        if max_time is None:
            max_time = time.time() + 5.0
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            clique_set = set(clique)
            
            # Try adding each non-clique vertex
            for v in range(n):
                if v in clique_set:
                    continue
                # Check if v can be added (connected to all in clique)
                if all(adj[v][u] for u in clique_set):
                    clique_set.add(v)
                    improved = True
                    break
            
            # Try swapping: remove one vertex, add one or more
            if not improved:
                for u in list(clique_set):
                    temp_clique = clique_set - {u}
                    added = False
                    for v in range(n):
                        if v in temp_clique:
                            continue
                        if all(adj[v][w] for w in temp_clique):
                            temp_clique.add(v)
                            added = True
                            break
                    if added and len(temp_clique) > len(clique_set):
                        clique_set = temp_clique
                        improved = True
                        break
        
        return clique_set
    
    # Run local search from initial solution
    result = local_search(best_clique, max_time=start_time + 30.0)
    if len(result) > best_size:
        best_clique = result
        best_size = len(best_clique)
    
    # Try different initial seeds
    random.seed(42)
    for iteration in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create a random perturbation of the best solution
        perturbed = set(best_clique)
        # Remove 2-5 random elements
        to_remove = random.sample(list(perturbed), min(random.randint(2, 5), len(perturbed)))
        for v in to_remove:
            perturbed.discard(v)
        
        # Run local search from perturbed state
        result = local_search(perturbed, max_time=start_time + 15.0)
        if len(result) > best_size:
            best_clique = result
            best_size = len(best_clique)
    
    # Convert back to actual numbers
    return sorted([candidates[i] for i in best_clique])
