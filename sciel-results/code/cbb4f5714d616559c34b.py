import random
import time

def construct():
    d = 7
    n = 2 ** d
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Pre-compute the "other 3 edges" for each face involving an edge?
    # For an edge (u, v) with diff bit k, the faces are defined by j != k.
    # The other edges are (u, u^2^j), (v, v^2^j), and ((u^2^j), (v^2^j)).
    # Let's precompute these triples for each edge to avoid bit operations in the loop.
    
    edge_info = {}
    for u, v in all_edges:
        # find k
        diff = u ^ v
        k = diff.bit_length() - 1
        faces = []
        for j in range(d):
            if j == k:
                continue
            c = u ^ (1 << j)
            e = v ^ (1 << j)
            # Edges in this face: (u,v), (u,c), (v,e), (c,e)
            # If we add (u,v), we need (u,c), (v,e), (c,e) to NOT all be present.
            # So we check if all 3 are in the set.
            e1 = (min(u, c), max(u, c))
            e2 = (min(v, e), max(v, e))
            e3 = (min(c, e), max(c, e))
            faces.append((e1, e2, e3))
        edge_info[(u, v)] = faces

    def can_add(edge, current_set):
        faces = edge_info[edge]
        for e1, e2, e3 in faces:
            if e1 in current_set and e2 in current_set and e3 in current_set:
                return False
        return True

    best_edges = []
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    random.seed(42) # Reproducibility for debugging, but we want randomness
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        current_set = set()
        current_list = []
        
        for edge in all_edges:
            if can_add(edge, current_set):
                current_set.add(edge)
                current_list.append(edge)
                
        if len(current_list) > best_count:
            best_count = len(current_list)
            best_edges = current_list
            
    return [[u, v] for u, v in best_edges]
