import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # Build edge index mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    
    # All K4s
    all_k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(all_k4s)
    
    # For each K4, list of edge indices
    k4_edge_idxs = []
    for k4 in all_k4s:
        e_idxs = []
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = k4[a], k4[b]
                if i > j:
                    i, j = j, i
                e_idxs.append(edge_to_idx[(i, j)])
        k4_edge_idxs.append(e_idxs)
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for ei in k4_edge_idxs[k4_idx]:
            edge_to_k4s[ei].append(k4_idx)
    
    def count_bad_and_colors(colors_arr):
        bad = 0
        color_set = set()
        for e_idxs in k4_edge_idxs:
            k4_colors = set()
            for ei in e_idxs:
                c = colors_arr[ei]
                k4_colors.add(c)
            color_set.update(k4_colors)
            if len(k4_colors) < 5:
                bad += 1
        return bad, len(color_set)
    
    def count_bad_for_k4s(colors_arr, k4_indices):
        bad = 0
        for idx in k4_indices:
            e_idxs = k4_edge_idxs[idx]
            k4_colors = set()
            for ei in e_idxs:
                k4_colors.add(colors_arr[ei])
            if len(k4_colors) < 5:
                bad += 1
        return bad
    
    best_result = None
    best_colors_used = 17  # seed value
    
    for target_K in [13, 14, 15]:
        if time.time() > deadline - 8:
            break
        
        num_restarts = 8 if target_K == 13 else 6
        for restart in range(num_restarts):
            if time.time() > deadline - 4:
                break
            
            # Random initialization
            colors_arr = np.random.randint(0, target_K, size=num_edges, dtype=np.int32)
            
            bad_init, cu_init = count_bad_and_colors(colors_arr)
            if bad_init == 0 and cu_init < best_colors_used:
                best_colors_used = cu_init
                best_result = colors_arr.copy()
                break
            
            # Simulated annealing with fast delta computation
            T = 2.0
            T_min = 0.01
            cooling = 0.9998
            max_iter = 30000
            iter_count = 0
            current_bad = bad_init
            best_local_bad = current_bad
            best_local_colors = colors_arr.copy()
            
            while iter_count < max_iter:
                if time.time() > deadline - 3:
                    break
                
                ei = np.random.randint(num_edges)
                old_color = colors_arr[ei]
                new_color = np.random.randint(target_K)
                if new_color == old_color:
                    new_color = (new_color + 1) % target_K
                
                # Compute delta: only K4s containing edge ei are affected
                affected_k4s = edge_to_k4s[ei]
                
                # Count bad in affected K4s before change
                old_bad_affected = 0
                for k4_idx in affected_k4s:
                    e_idxs = k4_edge_idxs[k4_idx]
                    k4_colors = set()
                    for eii in e_idxs:
                        k4_colors.add(colors_arr[eii])
                    if len(k4_colors) < 5:
                        old_bad_affected += 1
                
                # Apply change
                colors_arr[ei] = new_color
                iter_count += 1
                
                # Count bad in affected K4s after change
                new_bad_affected = 0
                for k4_idx in affected_k4s:
                    e_idxs = k4_edge_idxs[k4_idx]
                    k4_colors = set()
                    for eii in e_idxs:
                        k4_colors.add(colors_arr[eii])
                    if len(k4_colors) < 5:
                        new_bad_affected += 1
                
                delta = new_bad_affected - old_bad_affected
                if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    current_bad += delta
                    if current_bad < best_local_bad:
                        best_local_bad = current_bad
                        best_local_colors = colors_arr.copy()
                        if best_local_bad == 0:
                            break
                else:
                    colors_arr[ei] = old_color
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if best_local_bad == 0:
                _, cu = count_bad_and_colors(best_local_colors)
                if cu < best_colors_used:
                    best_colors_used = cu
                    best_result = best_local_colors.copy()
                break
    
    if best_result is not None:
        return list(best_result)
    
    # Fallback: seed coloring
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
