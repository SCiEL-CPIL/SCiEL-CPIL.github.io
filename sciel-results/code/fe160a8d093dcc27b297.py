import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_set = set()
    
    # Generate all edges
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_set.add((x, y))
                
    # Precompute faces for each edge
    # For each edge (u, v), find the faces it belongs to.
    # A face is a set of 4 edges.
    # If we add edge e, we must ensure that no face containing e has its other 3 edges already present.
    
    # Let's precompute for each edge, the list of other edges in the faces it belongs to.
    # Actually, for a given edge e=(u,v) along dimension i, the faces are determined by another dimension j.
    # The vertices of the face are u, v, u^j, v^j.
    # The edges are (u,v), (u^j, v^j), (u, u^j), (v, v^j).
    # So if we add (u,v), we must check for each j != i:
    #   if (u^j, v^j), (u, u^j), (v, v^j) are all in the current set, then we cannot add (u,v).
    
    # To speed up, we can precompute for each edge e, the list of "blocking triples".
    # But checking 6 triples of 3 edges each is fast enough.
    
    best_edges = set()
    best_count = 0
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    # Initial greedy to get a baseline
    current_edges = set()
    for e in edges:
        u, v = e
        # Find dimension i
        diff = u ^ v
        i = diff.bit_length() - 1
        ok = True
        for j in range(d):
            if j == i:
                continue
            # Check the other 3 edges of the face
            u_j = u ^ (1 << j)
            v_j = v ^ (1 << j)
            if ((min(u_j, v_j), max(u_j, v_j)) in current_edges and
                (min(u, u_j), max(u, u_j)) in current_edges and
                (min(v, v_j), max(v, v_j)) in current_edges):
                ok = False
                break
        if ok:
            current_edges.add(e)
            
    best_count = len(current_edges)
    best_edges = set(current_edges)
    
    # Randomized greedy restarts
    while time.time() - start_time < timeout:
        current_edges = set()
        random.shuffle(edges)
        for e in edges:
            u, v = e
            diff = u ^ v
            i = diff.bit_length() - 1
            ok = True
            for j in range(d):
                if j == i:
                    continue
                u_j = u ^ (1 << j)
                v_j = v ^ (1 << j)
                if ((min(u_j, v_j), max(u_j, v_j)) in current_edges and
                    (min(u, u_j), max(u, u_j)) in current_edges and
                    (min(v, v_j), max(v, v_j)) in current_edges):
                    ok = False
                    break
            if ok:
                current_edges.add(e)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    return [[u, v] for u, v in best_edges]
