# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_n = 39
    best_edges = []
    
    for target_n in [47, 46]:
        min_deg_required = target_n - 39
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            random.seed(attempt)
            adj_mask = [0] * target_n
            degrees = [0] * target_n
            
            def add_edge(u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
            
            stuck_count = 0
            while time.time() < deadline:
                low_deg_verts = [i for i in range(target_n) if degrees[i] < min_deg_required]
                
                if not low_deg_verts:
                    break
                
                u = random.choice(low_deg_verts)
                
                all_mask = (1 << target_n) - 1
                all_mask &= ~(1 << u)
                non_neighbors_mask = all_mask & ~adj_mask[u]
                
                candidates = []
                m = non_neighbors_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    
                    common = adj_mask[u] & adj_mask[v]
                    if common.bit_count() < 2:
                        candidates.append(v)
                        
                if not candidates:
                    stuck_count += 1
                    if stuck_count > 10:
                        break
                    continue
                
                v = min(candidates, key=lambda x: degrees[x])
                
                add_edge(u, v)
                stuck_count = 0
                
            if min(degrees) >= min_deg_required:
                best_n = target_n
                best_edges = []
                for i in range(target_n):
                    m = adj_mask[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            best_edges.append([i, j])
                        m ^= lsb
                return {'n': best_n, 'edges': best_edges}
                
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
