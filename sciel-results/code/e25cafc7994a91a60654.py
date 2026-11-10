import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None
    
    # All edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
    
    num_edges = len(all_edges)
    
    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx
    
    # Precompute face edge indices for fast lookup
    # face_edge_indices[idx] is a list of lists of 3 indices
    face_edge_indices = [[] for _ in range(num_edges)]
    
    for idx, (u, v, i) in enumerate(all_edges):
        faces = []
        for j in range(d):
            if j == i:
                continue
            u2 = u ^ (1 << j)
            v2 = v ^ (1 << j)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_tuple_to_idx[e1]
            idx2 = edge_tuple_to_idx[e2]
            idx3 = edge_tuple_to_idx[e3]
            faces.append((idx1, idx2, idx3))
        face_edge_indices[idx] = faces

    def try_greedy(order_seed=None):
        rng = random.Random(order_seed)
        chosen = [False] * num_edges
        count = 0
        
        indices = list(range(num_edges))
        rng.shuffle(indices)
        
        for idx in indices:
            can_add = True
            for i1, i2, i3 in face_edge_indices[idx]:
                if chosen[i1] and chosen[i2] and chosen[i3]:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
        
        return count, chosen
    
    attempt = 0
    while time.time() < deadline:
        count, chosen = try_greedy(attempt)
        if count > best_count:
            best_count = count
            best_chosen = chosen
        attempt += 1
    
    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
