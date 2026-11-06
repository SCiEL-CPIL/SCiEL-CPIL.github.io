import random
import time
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def greedy_merge(edges, rng):
    paths = [[u, v] for u, v in edges]
    active_paths = [True] * len(paths)
    
    while True:
        v_map = defaultdict(list)
        for i, p in enumerate(paths):
            if not active_paths[i]: continue
            v_map[p[0]].append((i, 0))
            v_map[p[1]].append((i, 1))
            
        merged = False
        verts = list(v_map.keys())
        rng.shuffle(verts)
        
        for v in verts:
            lst = v_map[v]
            if len(lst) < 2: continue
            
            # Group by path index
            path_indices = {}
            for idx, pos in lst:
                if idx not in path_indices:
                    path_indices[idx] = pos
            
            if len(path_indices) < 2:
                continue
                
            indices = list(path_indices.keys())
            rng.shuffle(indices)
            
            # Try to merge first two
            p1_idx = indices[0]
            p2_idx = indices[1]
            
            p1 = paths[p1_idx]
            p2 = paths[p2_idx]
            
            pos1 = path_indices[p1_idx]
            pos2 = path_indices[p2_idx]
            
            # Orient
            p1_oriented = p1 if pos1 == 1 else p1[::-1]
            p2_oriented = p2 if pos2 == 0 else p2[::-1]
            
            # Check if they meet at v
            if p1_oriented[-1] != v or p2_oriented[0] != v:
                continue
                
            # Check vertex disjointness
            set1 = set(p1_oriented)
            set2 = set(p2_oriented)
            intersection = set1 & set2
            if intersection != {v}:
                continue
                
            # Merge
            new_path = p1_oriented + p2_oriented[1:]
            paths[p1_idx] = new_path
            active_paths[p2_idx] = False
            merged = True
            break
            
        if not merged:
            break
            
    return [p for i, p in enumerate(paths) if active_paths[i]]

def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    start_time = time.time()
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        best_paths = None
        best_count = float('inf')
        
        local_rng = random.Random(123)
        
        iterations = 100
        for i in range(iterations):
            shuffled_edges = edges[:]
            local_rng.shuffle(shuffled_edges)
            
            paths = greedy_merge(shuffled_edges, local_rng)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
            if time.time() - start_time > 110:
                break
                
        results.append(best_paths)
        
    return results
