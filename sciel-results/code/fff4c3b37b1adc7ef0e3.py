import random
import time

def construct():
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute C4-free colorings for K5 and K6
    # K6 1-factorization (5 colors, each a perfect matching)
    # Vertices 0..5
    # M0: 0-1, 2-3, 4-5
    # M1: 0-2, 1-4, 3-5
    # M2: 0-3, 1-5, 2-4
    # M3: 0-4, 1-3, 2-5
    # M4: 0-5, 1-2, 3-4
    
    # Store as dict (u,v) -> color for u<v
    intra_k6 = {}
    matchings = [
        [(0,1), (2,3), (4,5)],
        [(0,2), (1,4), (3,5)],
        [(0,3), (1,5), (2,4)],
        [(0,4), (1,3), (2,5)],
        [(0,5), (1,2), (3,4)]
    ]
    for c, m in enumerate(matchings):
        for u, v in m:
            if u > v: u, v = v, u
            intra_k6[(u,v)] = c
            
    # K5 is K6 with vertex 5 removed
    intra_k5 = {}
    for (u,v), c in intra_k6.items():
        if u < 5 and v < 5:
            intra_k5[(u,v)] = c

    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        edges = []
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_idx_map[(i, j)] = idx
                idx += 1
        m = len(edges)
        
        best_colors = None
        best_energy = float('inf')
        
        max_restarts = 5 if n == 26 else 3
        time_per_restart = (time_limit - (time.time() - start_time)) / max_restarts
        
        for restart in range(max_restarts):
            restart_start = time.time()
            restart_limit = restart_start + time_per_restart
            if n == 26 and restart == 0:
                restart_limit = start_time + time_limit * 0.8
            
            # 1. Partition vertices into 5 sets
            vertices = list(range(n))
            random.shuffle(vertices)
            sets = [[] for _ in range(5)]
            for v in vertices:
                # Balance sizes
                min_set = min(sets, key=len)
                min_set.append(v)
            
            # Map global vertex to (set_id, local_index)
            v_to_set = {}
            for s_id, s in enumerate(sets):
                for l_idx, v in enumerate(s):
                    v_to_set[v] = (s_id, l_idx)
            
            # 2. Initialize colors
            colors = [0] * m
            adj = [ [set() for _ in range(n)] for _ in range(5) ]
            
            for k, (i, j) in enumerate(edges):
                s_i, l_i = v_to_set[i]
                s_j, l_j = v_to_set[j]
                
                if s_i == s_j:
                    # Intra-set: use precomputed C4-free coloring
                    # Ensure local indices are sorted for lookup
                    if l_i > l_j:
                        l_i, l_j = l_j, l_i
                    # Lookup in intra_k5 or intra_k6 based on set size
                    size = len(sets[s_i])
                    if size == 5:
                        c = intra_k5.get((l_i, l_j), 0)
                    elif size == 6:
                        c = intra_k6.get((l_i, l_j), 0)
                    else:
                        # Fallback for other sizes (should not happen for n<=26 balanced)
                        c = random.randint(0, 4)
                    colors[k] = c
                else:
                    # Inter-set: Random initialization (or Hash)
                    # Using random to allow SA to explore
                    c = random.randint(0, 4)
                    colors[k] = c
                
                # Build adjacency
                c = colors[k]
                adj[c][i].add(j)
                adj[c][j].add(i)
                
            def count_c4s_in_color(c):
                total = 0
                for u in range(n):
                    for v in adj[c][u]:
                        if u < v:
                            count = 0
                            for x in adj[c][v]:
                                if x == u: continue
                                for y in adj[c][u]:
                                    if y == v: continue
                                    if x == y: continue
                                    if y in adj[c][x]:
                                        count += 1
                            total += count
                return total // 4

            total_energy = sum(count_c4s_in_color(c) for c in range(5))
            
            if total_energy < best_energy:
                best_energy = total_energy
                best_colors = colors.copy()
                if best_energy == 0:
                    return {'n': n, 'colors': best_colors}

            T = 10.0
            cooling = 0.9995
            min_T = 0.01
            
            steps = 0
            max_steps = 100000
            
            while T > min_T:
                if time.time() > restart_limit:
                    break
                    
                k = random.randint(0, m - 1)
                i, j = edges[k]
                
                c_old = colors[k]
                c_new = random.randint(0, 4)
                if c_old == c_new: continue
                
                # Delta calc
                cnt_old = 0
                for x in adj[c_old][j]:
                    if x == i: continue
                    for y in adj[c_old][i]:
                        if y == j: continue
                        if x == y: continue
                        if y in adj[c_old][x]:
                            cnt_old += 1
                            
                cnt_new = 0
                for x in adj[c_new][j]:
                    if x == i: continue
                    for y in adj[c_new][i]:
                        if y == j: continue
                        if x == y: continue
                        if y in adj[c_new][x]:
                            cnt_new += 1
                            
                delta_E = cnt_new - cnt_old
                
                if delta_E < 0 or random.random() < pow(2.718, -delta_E / T):
                    colors[k] = c_new
                    adj[c_old][i].remove(j)
                    adj[c_old][j].remove(i)
                    adj[c_new][i].add(j)
                    adj[c_new][j].add(i)
                    
                    total_energy += delta_E
                    
                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_colors = colors.copy()
                        if best_energy == 0:
                            return {'n': n, 'colors': best_colors}
                            
                T *= cooling
                steps += 1
                
            if best_energy == 0:
                return {'n': n, 'colors': best_colors}
                
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
