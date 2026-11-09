import time
import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = d * (1 << (d - 1))
    
    edges = []
    edge_map = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                eid = len(edges)
                edges.append((x, y))
                edge_map[(x, y)] = eid
                edge_map[(y, x)] = eid
                
    # Precompute face neighbors
    # face_neighbors[eid] = list of 6 lists, each containing 3 edge IDs
    face_neighbors = [[] for _ in range(n_edges_total)]
    
    for eid, (u, v) in enumerate(edges):
        # Determine bit index
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            u_j = u ^ (1 << j)
            v_j = v ^ (1 << j)
            
            # Edges: (u, u_j), (v, v_j), (u_j, v_j)
            e1 = edge_map.get((min(u, u_j), max(u, u_j)))
            e2 = edge_map.get((min(v, v_j), max(v, v_j)))
            e3 = edge_map.get((min(u_j, v_j), max(u_j, v_j)))
            
            face_neighbors[eid].append([e1, e2, e3])

    def is_valid_addition(eid, selected):
        for face in face_neighbors[eid]:
            if all(e in selected for e in face):
                return False
        return True

    def greedy_init():
        selected = set()
        order = list(range(n_edges_total))
        random.shuffle(order)
        for eid in order:
            if is_valid_addition(eid, selected):
                selected.add(eid)
        return selected

    def local_search(selected):
        improved = True
        while improved:
            improved = False
            addable = []
            for eid in range(n_edges_total):
                if eid not in selected and is_valid_addition(eid, selected):
                    addable.append(eid)
            
            if addable:
                eid = random.choice(addable)
                selected.add(eid)
                improved = True
        
        return selected

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_selected = greedy_init()
    best_selected = local_search(best_selected)
    best_size = len(best_selected)
    
    while time.time() < deadline:
        selected = greedy_init()
        selected = local_search(selected)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected

    result = []
    for eid in best_selected:
        u, v = edges[eid]
        result.append([u, v])
    return result
