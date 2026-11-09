import time
import random
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def solve_n(n, time_limit):
        if n < 6:
            return [0] * (n * (n - 1) // 2)
        
        num_edges = n * (n - 1) // 2
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        # Generate C6 cycles
        cycle_edges = []
        
        for vertices in itertools.combinations(range(n), 6):
            v0 = vertices[0]
            others = vertices[1:]
            for perm in itertools.permutations(others):
                v1, v2, v3, v4, v5 = perm
                if v1 > v5:
                    continue
                c_edges = []
                cycle_verts = [v0, v1, v2, v3, v4, v5]
                for k in range(6):
                    u = cycle_verts[k]
                    v = cycle_verts[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    c_edges.append(edge_map[(u, v)])
                cycle_edges.append(c_edges)
        
        num_cycles = len(cycle_edges)
        
        # Build edge_to_cycles
        edge_to_cycles = [[] for _ in range(num_edges)]
        for c_idx, c_edges in enumerate(cycle_edges):
            for e_idx in c_edges:
                edge_to_cycles[e_idx].append(c_idx)
        
        # Convert to NumPy arrays
        edge_to_cycles_np = [np.array(c, dtype=np.int32) if c else np.array([], dtype=np.int32) for c in edge_to_cycles]
        
        best_mono = float('inf')
        best_colors = None
        
        max_restarts = 200
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
            
            cycle_counts = np.zeros((num_cycles, 4), dtype=np.int8)
            
            for c_idx, c_edges in enumerate(cycle_edges):
                for e_idx in c_edges:
                    cycle_counts[c_idx, colors[e_idx]] += 1
            
            mono_count = np.sum(np.any(cycle_counts == 6, axis=1))
            
            if mono_count == 0:
                return colors.tolist()
            
            if mono_count < best_mono:
                best_mono = mono_count
                best_colors = colors.copy()
            
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            steps_since_improve = 0
            
            while time.time() < time_limit:
                if mono_count == 0:
                    return colors.tolist()
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                
                best_new_color = -1
                best_delta = float('inf')
                
                idxs = edge_to_cycles_np[e_idx]
                if len(idxs) == 0:
                    # Edge not in any C6? Impossible for n>=6?
                    # In K6, every edge is in a C6?
                    # K6 has 15 edges. C6s use 6 edges.
                    # Yes, every edge is in some C6.
                    continue
                    
                counts_subset = cycle_counts[idxs]
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    old_mono = np.sum(counts_subset[:, old_color] == 6)
                    new_mono = np.sum(counts_subset[:, new_color] == 5)
                    
                    delta = new_mono - old_mono
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                
                if best_new_color == -1:
                    break
                    
                if best_delta < 0:
                    accept = True
                else:
                    if random.random() < math.exp(-best_delta / T):
                        accept = True
                    else:
                        accept = False
                
                if accept:
                    new_color = best_new_color
                    colors[e_idx] = new_color
                    mono_count += best_delta
                    
                    cycle_counts[idxs, old_color] -= 1
                    cycle_counts[idxs, new_color] += 1
                    
                    if mono_count < best_mono:
                        best_mono = mono_count
                        best_colors = colors.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                        
                    T = max(T_min, T * cooling)
                    
                    if steps_since_improve > 1000:
                        break
                else:
                    steps_since_improve += 1
                    T = max(T_min, T * cooling)
                    if steps_since_improve > 1000:
                        break
            
        if best_colors is not None:
            return best_colors.tolist()
        return None

    for n in [19, 18, 17, 16]:
        if time.time() > deadline - 5:
            break
            
        if n >= 18:
            alloc = 20
        elif n == 17:
            alloc = 30
        else:
            alloc = 10
            
        end_time = time.time() + alloc
        if end_time > deadline - 5:
            end_time = deadline - 5
            
        res = solve_n(n, end_time)
        if res is not None:
            return {'n': n, 'colors': res}
            
    n = 10
    colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    return {'n': n, 'colors': colors}
