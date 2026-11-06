import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 20
    best_edges = [[0, v, 0] for v in range(1, 20)]  # star fallback
    best_count = 19
    
    all_edges = list(combinations(range(n), 2))
    
    def is_valid(adj, colors, u, v, c):
        # Check if adding edge (u,v) with color c creates a non-rainbow 4-cycle
        # 4-cycle: u-a-b-v-u
        # Edges: (u,v), (u,a), (a,b), (b,v)
        # Colors: c, colors[(u,a)], colors[(a,b)], colors[(b,v)]
        
        for a in range(n):
            if a == u or a == v:
                continue
            if a not in adj[u]:
                continue
            ca = colors.get((u, a)) or colors.get((a, u))
            if ca is None:
                continue
            if ca == c:
                # If ca == c, then (u,v) and (u,a) share color.
                # They are adjacent, so they can't be in a 4-cycle together as opposite edges?
                # Wait, in a 4-cycle u-a-b-v-u, (u,v) and (u,a) are adjacent edges.
                # The condition is that ALL 4 edges have distinct colors.
                # So if ca == c, it's invalid immediately if a-b-v path exists.
                pass
            
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if b not in adj[a]:
                    continue
                if v not in adj[b]:
                    continue
                
                cab = colors.get((a, b)) or colors.get((b, a))
                cbv = colors.get((b, v)) or colors.get((v, b))
                
                if cab is None or cbv is None:
                    continue
                
                cols = [c, ca, cab, cbv]
                if len(set(cols)) != 4:
                    return False
        
        # Also check the other orientation? 
        # u-a-b-v-u is the same set of edges as v-b-a-u-v.
        # The check above covers all 4-cycles containing (u,v) because 
        # any 4-cycle containing (u,v) can be written as u-a-b-v-u for some a,b.
        
        return True

    for _ in range(2000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        adj = {i: set() for i in range(n)}
        colors = {}
        current_edges = []
        
        for u, v in all_edges:
            # Try to find a valid color
            found = False
            # Randomize color order to escape local minima
            color_order = list(range(n))
            random.shuffle(color_order)
            
            for c in color_order:
                if is_valid(adj, colors, u, v, c):
                    # Add edge
                    adj[u].add(v)
                    adj[v].add(u)
                    colors[(u, v)] = c
                    colors[(v, u)] = c
                    current_edges.append([u, v, c])
                    found = True
                    break
            
            if not found:
                # Skip this edge
                pass
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges

    return best_edges
