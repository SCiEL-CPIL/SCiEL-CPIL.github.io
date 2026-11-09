import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    k4_to_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    # k4_edge_pos[k4_id][edge_id] = position in the 6-edge list
    # Since num_edges is 120, we can use a dict or a 2D array of size (num_k4s, num_edges) with -1 for not present.
    # num_k4s = 1820. 1820 * 120 is small.
    
    edge_index = np.zeros((n, n), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[i, j] = idx
            edge_index[j, i] = idx
            idx += 1
            
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    edges = [
                        edge_index[i, j],
                        edge_index[i, k],
                        edge_index[i, l],
                        edge_index[j, k],
                        edge_index[j, l],
                        edge_index[k, l]
                    ]
                    k4_id = len(k4_to_edges)
                    k4_to_edges.append(edges)
                    for p, e in enumerate(edges):
                        edge_to_k4s[e].append(k4_id)
                    # Store position
                    # We'll use a dict of dicts or a list of dicts
                    # Actually, we can just store the list of (k4_id, pos) for each edge
                    pass

    # Precompute edge_to_k4_pos: list of lists of (k4_id, pos)
    edge_to_k4_pos = [[] for _ in range(num_edges)]
    for k4_id, edges in enumerate(k4_to_edges):
        for p, e in enumerate(edges):
            edge_to_k4_pos[e].append((k4_id, p))

    def count_bad_k4s(colors):
        bad_count = 0
        for edges in k4_to_edges:
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                bad_count += 1
        return bad_count

    def evaluate_delta(edge_idx, old_color, new_color, colors):
        if old_color == new_color:
            return 0
            
        delta = 0
        for k4_id, pos in edge_to_k4_pos[edge_idx]:
            edges = k4_to_edges[k4_id]
            c = [colors[e] for e in edges]
            
            old_set = set(c)
            old_bad = 1 if len(old_set) < 5 else 0
            
            c[pos] = new_color
            new_set = set(c)
            new_bad = 1 if len(new_set) < 5 else 0
            
            delta += new_bad - old_bad
            
        return delta

    best_colors = None
    best_bad = float('inf')
    
    K = 15
    colors = np.random.randint(0, K, size=num_edges)
    current_bad = count_bad_k4s(colors)
    
    if current_bad == 0:
        best_colors = colors.copy()
        best_bad = 0
    else:
        T = 1.0
        T_min = 0.01
        alpha = 0.99
        
        while time.time() < deadline - 5.0:
            if current_bad == 0:
                best_colors = colors.copy()
                best_bad = 0
                break
                
            e_idx = np.random.randint(0, num_edges)
            old_c = colors[e_idx]
            new_c = np.random.randint(0, K)
            
            if old_c == new_c:
                continue
                
            delta = evaluate_delta(e_idx, old_c, new_c, colors)
            
            if delta < 0:
                colors[e_idx] = new_c
                current_bad += delta
                if current_bad < best_bad:
                    best_bad = current_bad
                    if current_bad == 0:
                        best_colors = colors.copy()
                        break
            else:
                if delta == 0:
                    if random.random() < 0.5:
                        colors[e_idx] = new_c
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[e_idx] = new_c
                        current_bad += delta
                        
            T = max(T * alpha, T_min)
            
            if T == T_min and current_bad > 0:
                colors = np.random.randint(0, K, size=num_edges)
                current_bad = count_bad_k4s(colors)
                T = 1.0
                if current_bad < best_bad:
                    best_bad = current_bad
                    if current_bad == 0:
                        best_colors = colors.copy()
                        break

    if best_bad > 0:
        K = 16
        colors = np.random.randint(0, K, size=num_edges)
        current_bad = count_bad_k4s(colors)
        
        if current_bad == 0:
            best_colors = colors.copy()
            best_bad = 0
        else:
            T = 1.0
            T_min = 0.01
            alpha = 0.99
            end_time = time.time() + 10.0
            
            while time.time() < end_time:
                if current_bad == 0:
                    best_colors = colors.copy()
                    best_bad = 0
                    break
                    
                e_idx = np.random.randint(0, num_edges)
                old_c = colors[e_idx]
                new_c = np.random.randint(0, K)
                
                if old_c == new_c:
                    continue
                    
                delta = evaluate_delta(e_idx, old_c, new_c, colors)
                
                if delta < 0:
                    colors[e_idx] = new_c
                    current_bad += delta
                    if current_bad < best_bad:
                        best_bad = current_bad
                        if current_bad == 0:
                            best_colors = colors.copy()
                            break
                else:
                    if delta == 0:
                        if random.random() < 0.5:
                            colors[e_idx] = new_c
                    else:
                        if random.random() < np.exp(-delta / T):
                            colors[e_idx] = new_c
                            current_bad += delta
                            
                T = max(T * alpha, T_min)
                if T == T_min and current_bad > 0:
                    colors = np.random.randint(0, K, size=num_edges)
                    current_bad = count_bad_k4s(colors)
                    T = 1.0
                    if current_bad < best_bad:
                        best_bad = current_bad
                        if current_bad == 0:
                            best_colors = colors.copy()
                            break

    if best_colors is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    else:
        return [int(c) for c in best_colors]
