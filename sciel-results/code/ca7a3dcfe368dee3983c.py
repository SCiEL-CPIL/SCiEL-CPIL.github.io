import random
import time
import itertools

def construct():
    n = 20
    max_colors = 20
    vertices = list(range(n))
    
    # Map edge to index
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_to_idx[(u,v)] = idx
            idx_to_edge.append((u,v))
            idx += 1
            
    num_edges_possible = idx # 190
    
    # Precompute C4s as lists of edge indices
    # A C4 is defined by 4 vertices u,v,w,x.
    # Edges: (u,v), (v,w), (w,x), (x,u)
    # We need to ensure we capture all distinct C4s.
    # As before, 3 per 4-set.
    
    all_c4s = []
    for quad in itertools.combinations(vertices, 4):
        a,b,c,d = quad
        # C4 1: a-b-c-d-a
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,c)]
        e3 = edge_to_idx[(c,d)]
        e4 = edge_to_idx[(d,a)]
        all_c4s.append((e1, e2, e3, e4))
        
        # C4 2: a-b-d-c-a
        e1 = edge_to_idx[(a,b)]
        e2 = edge_to_idx[(b,d)]
        e3 = edge_to_idx[(d,c)]
        e4 = edge_to_idx[(c,a)]
        all_c4s.append((e1, e2, e3, e4))
        
        # C4 3: a-c-b-d-a
        e1 = edge_to_idx[(a,c)]
        e2 = edge_to_idx[(c,b)]
        e3 = edge_to_idx[(b,d)]
        e4 = edge_to_idx[(d,a)]
        all_c4s.append((e1, e2, e3, e4))
        
    num_c4s = len(all_c4s)
    
    # State
    present = [0] * num_edges_possible
    colors = [0] * num_edges_possible
    
    # Initialize
    random.seed(42)
    for i in range(num_edges_possible):
        if random.random() < 0.5:
            present[i] = 1
            colors[i] = random.randint(0, max_colors-1)
            
    def calc_penalty():
        pen = 0
        for c4 in all_c4s:
            e1, e2, e3, e4 = c4
            if present[e1] and present[e2] and present[e3] and present[e4]:
                c1, c2, c3, c4_col = colors[e1], colors[e2], colors[e3], colors[e4]
                if c1 == c2 or c1 == c3 or c1 == c4_col or c2 == c3 or c2 == c4_col or c3 == c4_col:
                    pen += 1
        return pen

    current_pen = calc_penalty()
    best_pen = current_pen
    best_present = present[:]
    best_colors = colors[:]
    
    start_time = time.time()
    timeout = 110.0
    T = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        # Move: Recolor random edge
        i = random.randint(0, num_edges_possible-1)
        if present[i]:
            old_c = colors[i]
            new_c = random.randint(0, max_colors-1)
            if new_c != old_c:
                colors[i] = new_c
                new_pen = calc_penalty()
                delta = new_pen - current_pen
                if delta < 0 or random.random() < pow(2.718, -delta/T):
                    current_pen = new_pen
                    if current_pen < best_pen:
                        best_pen = current_pen
                        best_present = present[:]
                        best_colors = colors[:]
                else:
                    colors[i] = old_c
        else:
            # Try to add edge
            new_c = random.randint(0, max_colors-1)
            present[i] = 1
            colors[i] = new_c
            new_pen = calc_penalty()
            delta = new_pen - current_pen
            # We want to add edges, so maybe bias towards adding if delta is small?
            # But if delta > 0, it's a penalty.
            if delta < 0 or (delta == 0 and random.random() < 0.5) or random.random() < pow(2.718, -delta/T):
                current_pen = new_pen
                if current_pen < best_pen:
                    best_pen = current_pen
                    best_present = present[:]
                    best_colors = colors[:]
            else:
                present[i] = 0
                colors[i] = old_c if 'old_c' in locals() else 0 # Cleanup
                
        # Occasionally try to remove an edge if penalty is high
        if current_pen > 0 and random.random() < 0.1:
            i = random.randint(0, num_edges_possible-1)
            if present[i]:
                present[i] = 0
                new_pen = calc_penalty()
                if new_pen < current_pen:
                    current_pen = new_pen
                    if current_pen < best_pen:
                        best_pen = current_pen
                        best_present = present[:]
                        best_colors = colors[:]
                else:
                    present[i] = 1
                    
        T *= cooling

    # Reconstruct result
    result = []
    for i in range(num_edges_possible):
        if best_present[i]:
            u, v = idx_to_edge[i]
            result.append([u, v, best_colors[i]])
            
    return result
