import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    best_edges = []
    max_edges = 0
    
    n = 20
    colors = list(range(n))
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Helper to get color
    # We will use a dict for O(1) lookups
    
    while time.time() < deadline:
        # Reset
        edge_colors = {} # key: (min_u, max_v), value: color
        current_edges = []
        
        # Shuffle edges to randomize insertion order
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            key = (min(u, v), max(u, v))
            if key in edge_colors:
                continue # Already added, though loop is unique, just in case
            
            forbidden = set()
            blocked = False
            
            # Check all paths u-a-b-v
            # a, b must be distinct from u, v and each other
            for a in range(n):
                if a == u or a == v:
                    continue
                key_ua = (min(u, a), max(u, a))
                if key_ua not in edge_colors:
                    continue
                c_ua = edge_colors[key_ua]
                
                for b in range(n):
                    if b == u or b == v or b == a:
                        continue
                    key_ab = (min(a, b), max(a, b))
                    key_bv = (min(b, v), max(b, v))
                    
                    if key_ab not in edge_colors or key_bv not in edge_colors:
                        continue
                        
                    c_ab = edge_colors[key_ab]
                    c_bv = edge_colors[key_bv]
                    
                    # The colors of the path edges must be distinct
                    # Otherwise, the 4-cycle u-a-b-v-u would have duplicate colors
                    # regardless of the new edge's color
                    if len({c_ua, c_ab, c_bv}) < 3:
                        blocked = True
                        break
                    
                    forbidden.update([c_ua, c_ab, c_bv])
                        
                if blocked:
                    break
                
            if blocked:
                continue
            
            # Pick a color not in forbidden
            available = [c for c in colors if c not in forbidden]
            if not available:
                continue
            
            c = random.choice(available)
            edge_colors[key] = c
            current_edges.append([u, v, c])
            
        if len(current_edges) > max_edges:
            max_edges = len(current_edges)
            best_edges = current_edges[:]
            
    return best_edges
