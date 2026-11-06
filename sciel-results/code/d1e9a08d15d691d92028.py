import numpy as np
import time
import itertools
import random

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Precompute all triangles
    tri_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_list.append((i, j, k))
    tri_arr = np.array(tri_list)
    I = tri_arr[:, 0]
    J = tri_arr[:, 1]
    K = tri_arr[:, 2]
    n_tri = len(tri_arr)
    
    A = list(range(8))
    B = list(range(8, 16))
    all_cross = [(u, v) for u in A for v in B]
    all_int_A = list(itertools.combinations(A, 2))
    all_int_B = list(itertools.combinations(B, 2))
    
    def get_score_fast(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return int(np.max(sums))
    
    def build_adj(removed_cross, int_A, int_B):
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        for u, v in removed_cross:
            adj[u, v] = 0
            adj[v, u] = 0
        for u, v in int_A:
            adj[u, v] = 1
            adj[v, u] = 1
        for u, v in int_B:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        return non_edges
    
    def find_worst_triangle(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return None, 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        max_idx = np.argmax(sums)
        valid_indices = np.where(valid)[0]
        tri_idx = valid_indices[max_idx]
        return (I[tri_idx], J[tri_idx], K[tri_idx]), int(sums[max_idx])
    
    # Phase 1: Systematic search over small perturbations of K_{8,8}
    best_adj = None
    best_score = 999
    
    # k=0: baseline (64 cross + 1 internal = 65)
    for u, v in all_int_A:
        adj = build_adj([], [(u, v)], [])
        if adj.sum() // 2 == target_edges:
            s = get_score_fast(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()
    
    for u, v in all_int_B:
        adj = build_adj([], [], [(u, v)])
        if adj.sum() // 2 == target_edges:
            s = get_score_fast(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()
    
    # k=1: remove 1 cross, add 2 internal
    if time.time() - start_time < deadline - 50:
        for rc in all_cross:
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj([rc], [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score_fast(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            for ia1, ia2 in itertools.combinations(all_int_A, 2):
                adj = build_adj([rc], [ia1, ia2], [])
                if adj.sum() // 2 == target_edges:
                    s = get_score_fast(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            for ib1, ib2 in itertools.combinations(all_int_B, 2):
                adj = build_adj([rc], [], [ib1, ib2])
                if adj.sum() // 2 == target_edges:
                    s = get_score_fast(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            if time.time() - start_time > deadline - 50:
                break
    
    # k=2: remove 2 cross, add 3 internal (limited search)
    if time.time() - start_time < deadline - 30:
        cross_sample = random.sample(all_cross, min(20, len(all_cross)))
        for rc1, rc2 in itertools.combinations(cross_sample, 2):
            removed = [rc1, rc2]
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj(removed, [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score_fast(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            if time.time() - start_time > deadline - 30:
                break
    
    if best_adj is None:
        best_adj = build_adj([], [(0, 1)], [])
        best_score = get_score_fast(best_adj)
    
    # Phase 2: Simulated annealing with targeted moves
    rng = np.random.RandomState(42)
    
    sa_configs = [
        {'T0': 5.0, 'cooling': 0.9995, 'min_T': 0.1, 'steps': 80000},
        {'T0': 3.0, 'cooling': 0.9997, 'min_T': 0.2, 'steps': 60000},
        {'T0': 4.0, 'cooling': 0.9993, 'min_T': 0.15, 'steps': 70000},
    ]
    
    for cfg_idx, cfg in enumerate(sa_configs):
        if time.time() - start_time > deadline - 20:
            break
        
        # Start from best found so far or a perturbed version
        if cfg_idx == 0:
            current_adj = best_adj.copy()
        else:
            # Perturb the best solution
            current_adj = best_adj.copy()
            edges = get_edges(current_adj)
            non_edges = get_non_edges(current_adj)
            if edges and non_edges:
                for _ in range(3):
                    u, v = edges[rng.randint(len(edges))]
                    x, y = non_edges[rng.randint(len(non_edges))]
                    if x != u and x != v and y != u and y != v:
                        current_adj[u, v] = 0
                        current_adj[v, u] = 0
                        current_adj[x, y] = 1
                        current_adj[y, x] = 1
                        edges = get_edges(current_adj)
                        non_edges = get_non_edges(current_adj)
        
        current_score = get_score_fast(current_adj)
        T = cfg['T0']
        cooling = cfg['cooling']
        min_T = cfg['min_T']
        steps = cfg['steps']
        
        for step in range(steps):
            if time.time() - start_time > deadline - 20:
                break
            
            r = rng.random()
            
            if r < 0.5:
                # Targeted move: swap an edge incident to worst triangle
                worst_tri, worst_score = find_worst_triangle(current_adj)
                if worst_tri is None:
                    break
                
                u, v, w = worst_tri
                # Get edges incident to the worst triangle vertices
                incident_edges = []
                for vert in (u, v, w):
                    for other in range(n):
                        if other != vert and current_adj[vert, other] == 1:
                            if vert < other:
                                incident_edges.append((vert, other))
                            else:
                                incident_edges.append((other, vert))
                
                if not incident_edges:
                    continue
                
                remove_edge = incident_edges[rng.randint(len(incident_edges))]
                non_edges = get_non_edges(current_adj)
                if not non_edges:
                    continue
                
                add_edge = non_edges[rng.randint(len(non_edges))]
                
                # Ensure no vertex conflict
                ru, rv = remove_edge
                au, av = add_edge
                if au in (ru, rv) or av in (ru, rv):
                    continue
                
                current_adj[ru, rv] = 0
                current_adj[rv, ru] = 0
                current_adj[au, av] = 1
                current_adj[av, au] = 1
                
                new_score = get_score_fast(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                else:
                    delta = new_score - current_score
                    if rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                    else:
                        current_adj[ru, rv] = 1
                        current_adj[rv, ru] = 1
                        current_adj[au, av] = 0
                        current_adj[av, au] = 0
                        
            elif r < 0.8:
                # Random edge swap
                edges = get_edges(current_adj)
                non_edges = get_non_edges(current_adj)
                if not edges or not non_edges:
                    continue
                
                u, v = edges[rng.randint(len(edges))]
                x, y = non_edges[rng.randint(len(non_edges))]
                
                if x in (u, v) or y in (u, v):
                    continue
                
                current_adj[u, v] = 0
                current_adj[v, u] = 0
                current_adj[x, y] = 1
                current_adj[y, x] = 1
                
                new_score = get_score_fast(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                else:
                    delta = new_score - current_score
                    if rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                    else:
                        current_adj[u, v] = 1
                        current_adj[v, u] = 1
                        current_adj[x, y] = 0
                        current_adj[y, x] = 0
                        
            else:
                # 2-edge swap for more aggressive exploration
                edges = get_edges(current_adj)
                non_edges = get_non_edges(current_adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    continue
                
                idx1 = rng.randint(len(edges))
                idx2 = rng.randint(len(edges))
                if idx1 == idx2:
                    continue
                e1 = edges[idx1]
                e2 = edges[idx2]
                
                nidx1 = rng.randint(len(non_edges))
                nidx2 = rng.randint(len(non_edges))
                if nidx1 == nidx2:
                    continue
                ne1 = non_edges[nidx1]
                ne2 = non_edges[nidx2]
                
                # Check for vertex conflicts
                all_remove = set(e1 + e2)
                all_add = set(ne1 + ne2)
                if all_remove & all_add:
                    continue
                
                u1, v1 = e1
                u2, v2 = e2
                x1, y1 = ne1
                x2, y2 = ne2
                
                current_adj[u1, v1] = 0
                current_adj[v1, u1] = 0
                current_adj[u2, v2] = 0
                current_adj[v2, u2] = 0
                current_adj[x1, y1] = 1
                current_adj[y1, x1] = 1
                current_adj[x2, y2] = 1
                current_adj[y2, x2] = 1
                
                new_score = get_score_fast(current_adj)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj.copy()
                else:
                    delta = new_score - current_score
                    if rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                    else:
                        current_adj[u1, v1] = 1
                        current_adj[v1, u1] = 1
                        current_adj[u2, v2] = 1
                        current_adj[v2, u2] = 1
                        current_adj[x1, y1] = 0
                        current_adj[y1, x1] = 0
                        current_adj[x2, y2] = 0
                        current_adj[y2, x2] = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
    
    # Phase 3: Thorough local search polish
    if time.time() - start_time < deadline - 3:
        improved = True
        polish_round = 0
        while improved and time.time() - start_time < deadline - 3:
            improved = False
            polish_round += 1
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)
            
            # Try all single-edge swaps
            for u, v in edges:
                if time.time() - start_time > deadline - 3:
                    break
                for x, y in non_edges:
                    if x in (u, v) or y in (u, v):
                        continue
                    
                    best_adj[u, v] = 0
                    best_adj[v, u] = 0
                    best_adj[x, y] = 1
                    best_adj[y, x] = 1
                    
                    new_score = get_score_fast(best_adj)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        non_edges = get_non_edges(best_adj)
                    else:
                        best_adj[u, v] = 1
                        best_adj[v, u] = 1
                        best_adj[x, y] = 0
                        best_adj[y, x] = 0
            
            # If we improved, also try 2-edge swaps
            if improved and time.time() - start_time < deadline - 3:
                edges = get_edges(best_adj)
                non_edges = get_non_edges(best_adj)
                for i in range(len(edges)):
                    if time.time() - start_time > deadline - 3:
                        break
                    u1, v1 = edges[i]
                    for j in range(i+1, len(edges)):
                        if time.time() - start_time > deadline - 3:
                            break
                        u2, v2 = edges[j]
                        for k in range(len(non_edges)):
                            if time.time() - start_time > deadline - 3:
                                break
                            x1, y1 = non_edges[k]
                            if x1 in (u1, v1, u2, v2) or y1 in (u1, v1, u2, v2):
                                continue
                            for l in range(k+1, len(non_edges)):
                                if time.time() - start_time > deadline - 3:
                                    break
                                x2, y2 = non_edges[l]
                                if x2 in (u1, v1, u2, v2, x1, y1) or y2 in (u1, v1, u2, v2, x1, y1):
                                    continue
                                
                                best_adj[u1, v1] = 0
                                best_adj[v1, u1] = 0
                                best_adj[u2, v2] = 0
                                best_adj[v2, u2] = 0
                                best_adj[x1, y1] = 1
                                best_adj[y1, x1] = 1
                                best_adj[x2, y2] = 1
                                best_adj[y2, x2] = 1
                                
                                new_score = get_score_fast(best_adj)
                                if new_score < best_score:
                                    best_score = new_score
                                    improved = True
                                else:
                                    best_adj[u1, v1] = 1
                                    best_adj[v1, u1] = 1
                                    best_adj[u2, v2] = 1
                                    best_adj[v2, u2] = 1
                                    best_adj[x1, y1] = 0
                                    best_adj[y1, x1] = 0
                                    best_adj[x2, y2] = 0
                                    best_adj[y2, x2] = 0
    
    # Return result
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
    
    return {'n': n, 'edges': edges_list}
