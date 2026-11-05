import time
import random

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    best_edges = []
    best_count = 0
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def check_validity(u, v, c, graph):
        # graph: dict (u, v) -> color
        # We add (u, v) with color c.
        # Check all 4-cycles u-v-x-y-u
        # Edges involved: (u,v), (v,x), (x,y), (y,u)
        for x in range(n):
            if x == u or x == v:
                continue
            # Edge (v, x) must exist
            c_vx = graph.get((v, x))
            if c_vx is None:
                c_vx = graph.get((x, v))
            if c_vx is None:
                continue
            
            if c_vx == c:
                return False
            
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                
                # Edge (x, y) must exist
                c_xy = graph.get((x, y))
                if c_xy is None:
                    c_xy = graph.get((y, x))
                if c_xy is None:
                    continue
                
                if c_xy == c or c_xy == c_vx:
                    return False
                
                # Edge (y, u) must exist
                c_yu = graph.get((y, u))
                if c_yu is None:
                    c_yu = graph.get((u, y))
                if c_yu is None:
                    continue
                
                if c_yu == c or c_yu == c_vx or c_yu == c_xy:
                    return False
        
        return True

    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        graph = {}
        current_edges = []
        
        for u, v in all_edges:
            # Try to find a color for (u, v)
            # Randomize color order to avoid bias
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for c in colors:
                if check_validity(u, v, c, graph):
                    graph[(u, v)] = c
                    graph[(v, u)] = c
                    current_edges.append([u, v, c])
                    added = True
                    break
            
            if time.time() > deadline:
                break
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return best_edges
