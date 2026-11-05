import time
import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Generate all edges
    all_edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge = (u, v)
                idx = len(all_edges)
                all_edges.append(edge)
                edge_to_idx[edge] = idx
                
    # Pre-compute faces for each edge as sets of indices of other edges
    edge_faces_idx = [set() for _ in range(n_edges)]
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            u2 = u ^ mask
            v2 = v ^ mask
            
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            
            idx1 = edge_to_idx[e1]
            idx2 = edge_to_idx[e2]
            idx3 = edge_to_idx[e3]
            
            edge_faces_idx[idx].add((idx1, idx2, idx3))
            
    def can_add(idx, chosen):
        for triplet in edge_faces_idx[idx]:
            if triplet[0] in chosen and triplet[1] in chosen and triplet[2] in chosen:
                return False
        return True

    def run_greedy(random_seed):
        random.seed(random_seed)
        indices = list(range(n_edges))
        random.shuffle(indices)
        
        chosen = set()
        for idx in indices:
            if can_add(idx, chosen):
                chosen.add(idx)
        return chosen

    def local_search(chosen):
        chosen = set(chosen)
        improved = True
        # Limit iterations to prevent infinite loops if logic is flawed, though size increase ensures termination
        max_iters = 100 
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try removing each edge and adding as many as possible
            for e_to_remove in list(chosen):
                old_size = len(chosen)
                chosen.remove(e_to_remove)
                
                # Try to add edges not in chosen
                added = []
                for idx in range(n_edges):
                    if idx in chosen:
                        continue
                    if can_add(idx, chosen):
                        chosen.add(idx)
                        added.append(idx)
                        
                if len(chosen) > old_size:
                    improved = True
                else:
                    # Restore
                    chosen.add(e_to_remove)
                    for idx in added:
                        chosen.remove(idx)
                        
        return chosen

    best_size = 0
    best_set = set()
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    seed = 0
    while time.time() < deadline:
        chosen = run_greedy(seed)
        # Local search
        optimized = local_search(chosen)
        
        if len(optimized) > best_size:
            best_size = len(optimized)
            best_set = optimized
            
        seed += 1
        
    # Convert back to list of [u, v]
    result = []
    for idx in best_set:
        u, v = all_edges[idx]
        result.append([u, v])
        
    return result
