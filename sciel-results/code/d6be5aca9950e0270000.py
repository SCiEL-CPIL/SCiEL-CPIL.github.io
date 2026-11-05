"""Optimized greedy for rainbow 4-cycle edge coloring on 20 vertices."""
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    best_edges = None
    best_count = 0
    
    # Try multiple random restarts with different edge orderings
    random.seed(42)
    
    max_restarts = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Build candidate edge list
        candidate_edges = []
        for i in range(n):
            for j in range(i+1, n):
                candidate_edges.append((i, j))
        
        # Shuffle edge order
        random.shuffle(candidate_edges)
        
        # Greedy construction
        edge_list = []
        
        # Track adjacency for fast 4-cycle checking
        neighbors = {v: set() for v in range(n)}
        edge_color = {}  # frozenset({u,v}) -> color
        
        for (u, v) in candidate_edges:
            if time.time() - start_time > DEADLINE:
                break
            
            # Check if edge (u,v) can be added
            # A new 4-cycle involving (u,v) must have the form u-v-x-w-u
            # This requires a path of length 3 from u to v in the existing graph: u-w-x-v
            # The edges in the cycle are (u,v), (v,x), (x,w), (w,u)
            # We need these 4 edges to have distinct colors.
            # Since (u,v) is new, we choose its color.
            # The other 3 edges (v,x), (x,w), (w,u) are existing.
            # They must have distinct colors among themselves, and distinct from (u,v).
            
            forbidden = set()
            can_add = True
            
            # Iterate over all paths of length 3 from u to v
            # Path: u -> w -> x -> v
            # w is neighbor of u
            # x is neighbor of v
            # (w, x) must be an edge
            
            for w in neighbors[u]:
                if w == v:
                    continue
                for x in neighbors[v]:
                    if x == u:
                        continue
                    if x == w:
                        continue
                    
                    # Check if (w, x) is an edge
                    key_wx = frozenset((w, x))
                    if key_wx not in edge_color:
                        continue
                    
                    # We have a path u-w-x-v
                    # Get colors of the 3 existing edges
                    key_uw = frozenset((u, w))
                    key_xv = frozenset((x, v))
                    # key_wx is already checked
                    
                    c_1 = edge_color[key_uw]
                    c_2 = edge_color[key_wx]
                    c_3 = edge_color[key_xv]
                    
                    # Check if these 3 are distinct
                    if len({c_1, c_2, c_3}) < 3:
                        can_add = False
                        break
                
                if not can_add:
                    break
            
            if not can_add:
                continue
            
            # If we are here, all paths of length 3 from u to v have distinct edge colors.
            # Collect all colors involved in these paths to forbid them for (u,v)
            # Note: We must iterate again to collect colors if we didn't break early?
            # Actually, if can_add is True, we need the union of all colors from all valid paths.
            # The previous loop broke if invalid. If it didn't break, we need to re-scan or store.
            # Let's re-scan to be safe and simple, as n is small.
            
            forbidden = set()
            for w in neighbors[u]:
                if w == v:
                    continue
                for x in neighbors[v]:
                    if x == u or x == w:
                        continue
                    
                    key_wx = frozenset((w, x))
                    if key_wx not in edge_color:
                        continue
                    
                    key_uw = frozenset((u, w))
                    key_xv = frozenset((x, v))
                    
                    c_1 = edge_color[key_uw]
                    c_2 = edge_color[key_wx]
                    c_3 = edge_color[key_xv]
                    
                    forbidden.add(c_1)
                    forbidden.add(c_2)
                    forbidden.add(c_3)
            
            # Pick a color not in forbidden
            available = [c for c in colors if c not in forbidden]
            if not available:
                continue  # Skip this edge
            
            c = random.choice(available)
            key = frozenset((u, v))
            edge_color[key] = c
            edge_list.append([u, v, c])
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        count = len(edge_list)
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edge_list]
        
        # Early termination if we found a very good solution
        if best_count >= 70:
            break
    
    if best_edges is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, n)]
    
    return best_edges
