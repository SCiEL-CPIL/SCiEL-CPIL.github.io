import numpy as np
import time
from collections import deque

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Calculate forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency list for the conflict graph
    # Graph G: edge (u, v) if (u-v) % m in forbidden
    adj = [set() for _ in range(m)]
    for u in range(m):
        for f in forbidden:
            v = (u + f) % m
            if v > u: # Avoid duplicates, but for MIS we usually need symmetric adjacency
                adj[u].add(v)
                adj[v].add(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
                
    # Note: The above loop adds both directions. 
    # Let's just build it cleanly.
    adj = [set() for _ in range(m)]
    for u in range(m):
        for v in range(u+1, m):
            diff = (u - v) % m
            if diff in forbidden:
                adj[u].add(v)
                adj[v].add(u)
                
    best_size = 0
    best_set = set()
    
    # Initial greedy solution to get a lower bound
    # Greedy: pick vertex with min degree, remove it and neighbors
    remaining = list(range(m))
    greedy_set = set()
    rem = set(remaining)
    while rem:
        # Pick vertex with min degree in remaining graph
        min_deg = m
        min_v = -1
        for v in rem:
            deg = len(adj[v] & rem)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        greedy_set.add(min_v)
        rem.remove(min_v)
        rem -= adj[min_v]
        
    best_size = len(greedy_set)
    best_set = greedy_set.copy()
    
    # Branch and Bound for Maximum Independent Set
    # We use bitsets for speed if possible, but sets are okay for 205 nodes.
    # Let's use a recursive function.
    
    def branch_and_bound(candidates, current_set):
        nonlocal best_size, best_set
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return
            
        # Upper bound: current size + number of candidates
        if len(current_set) + len(candidates) <= best_size:
            return
            
        # Heuristic ordering: sort candidates by degree in the subgraph induced by candidates
        # High degree vertices are more likely to be excluded? Or low degree?
        # In MIS, we want to pick vertices. 
        # Branching on a vertex with high degree in the remaining graph is often effective 
        # because excluding it removes many constraints? 
        # Actually, standard MIS algorithms often branch on the vertex with the highest degree 
        # in the remaining graph.
        
        # Let's pick the vertex with the highest degree in the induced subgraph
        max_deg = -1
        pivot = -1
        for v in candidates:
            deg = len(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        # Branch 1: Include pivot
        new_candidates = candidates - adj[pivot] - {pivot}
        new_current = current_set | {pivot}
        branch_and_bound(new_candidates, new_current)
        
        # Branch 2: Exclude pivot
        new_candidates = candidates - {pivot}
        branch_and_bound(new_candidates, current_set)
        
    # Start with all vertices
    all_vertices = set(range(m))
    
    # Time limit guard
    start_time = time.time()
    TIME_LIMIT = 580  # Leave some margin for overhead
    
    # To avoid recursion depth issues and to allow time checking, 
    # we can implement this iteratively or with a recursion limit.
    # Python recursion limit is usually 1000, depth here is at most 205, so it's fine.
    import sys
    sys.setrecursionlimit(3000)
    
    # We need a way to abort if it takes too long.
    # The recursive function doesn't check time. 
    # We can wrap it or check time in the main loop if we restructure.
    # Given the size (205), B&B with greedy coloring bound is very fast usually.
    # Without coloring bound, it might be slow. 
    # Let's add a simple coloring bound.
    
    def greedy_coloring(candidates):
        # Returns number of colors used
        # This is an upper bound on the size of the independent set in candidates
        # because an independent set can contain at most one vertex of each color
        # in a proper coloring of the graph.
        # Wait, the bound is: alpha(G) <= chromatic_number(G).
        # Greedy coloring gives an upper bound on chromatic number.
        
        colors = {}
        uncolored = set(candidates)
        color_idx = 0
        while uncolored:
            color_idx += 1
            # Find a maximal independent set in the remaining graph
            # Vertices that are not adjacent to each other can share a color
            # We pick vertices greedily
            current_color_set = set()
            # Sort by degree descending? 
            # For coloring, we usually pick vertices with high degree first to constrain others?
            # Or just pick any.
            # Let's pick the one with highest degree in uncolored
            for v in sorted(list(uncolored), key=lambda x: len(adj[x] & uncolored), reverse=True):
                if v in uncolored:
                    # Check if v can take this color
                    # It can if no neighbor already has this color
                    # Since we are building the color set, we just need to ensure
                    # that v is not adjacent to any vertex in current_color_set
                    if not (adj[v] & current_color_set):
                        current_color_set.add(v)
                        uncolored.remove(v)
        return color_idx

    def branch_and_bound_v2(candidates, current_set):
        nonlocal best_size, best_set
        
        if time.time() - start_time > TIME_LIMIT:
            return False # Signal timeout
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return True
            
        # Bound
        upper_bound = len(current_set) + greedy_coloring(candidates)
        if upper_bound <= best_size:
            return True
            
        # Pivot selection: highest degree in induced subgraph
        max_deg = -1
        pivot = -1
        for v in candidates:
            deg = len(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                pivot = v
                
        # Branch 1: Include pivot
        new_candidates = candidates - adj[pivot] - {pivot}
        new_current = current_set | {pivot}
        if not branch_and_bound_v2(new_candidates, new_current):
            return False
            
        # Branch 2: Exclude pivot
        new_candidates = candidates - {pivot}
        if not branch_and_bound_v2(new_candidates, current_set):
            return False
            
        return True
        
    branch_and_bound_v2(all_vertices, set())
    
    return np.array(list(best_set), dtype=np.int64)
