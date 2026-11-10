import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    vertices = list(range(n))
    
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
            
    adj = [[False] * n for _ in range(n)]
    degrees = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            edges.add((u, v))
            adj[u][v] = True
            adj[v][u] = True
            degrees[u] += 1
            degrees[v] += 1
            
    subsets_8 = list(itertools.combinations(vertices, 8))
    num_subsets = len(subsets_8)
    
    edge_to_subsets = {e: [] for e in edges}
    for i, s in enumerate(subsets_8):
        s_set = set(s)
        for u in s:
            for v in s:
                if u < v:
                    if (u, v) in edge_to_subsets:
                        edge_to_subsets[(u, v)].append(i)
                elif v < u:
                    if (v, u) in edge_to_subsets:
                        edge_to_subsets[(v, u)].append(i)
    
    subset_edge_counts = [28] * num_subsets
    
    q3_edges = [(0, 4), (0, 5), (0, 6),
                (1, 4), (1, 5), (1, 7),
                (2, 4), (2, 6), (2, 7),
                (3, 5), (3, 6), (3, 7)]
    
    partitions = []
    for A in itertools.combinations(range(8), 4):
        A_set = set(A)
        B = tuple(v for v in range(8) if v not in A_set)
        perms_A = list(itertools.permutations(A))
        perms_B = list(itertools.permutations(B))
        partitions.append((A, B, perms_A, perms_B))
        
    while time.time() < deadline:
        found_q3 = False
        
        for idx in range(num_subsets):
            if time.time() > deadline:
                break
                
            if subset_edge_counts[idx] < 12:
                continue
                
            subset = subsets_8[idx]
            
            local_adj = [[False]*8 for _ in range(8)]
            for i in range(8):
                for j in range(i+1, 8):
                    u = subset[i]
                    v = subset[j]
                    if adj[u][v]:
                        local_adj[i][j] = True
                        local_adj[j][i] = True
            
            found_in_subset = False
            q3_edges_found = []
            
            for A, B, perms_A, perms_B in partitions:
                edge_count_ab = 0
                for u in A:
                    for v in B:
                        if local_adj[u][v]:
                            edge_count_ab += 1
                if edge_count_ab < 12:
                    continue
                
                for perm_A in perms_A:
                    for perm_B in perms_B:
                        valid = True
                        for u, v in q3_edges:
                            if not local_adj[perm_A[u]][perm_B[v-4]]:
                                valid = False
                                break
                        if valid:
                            found_in_subset = True
                            for u, v in q3_edges:
                                gu = subset[perm_A[u]]
                                gv = subset[perm_B[v-4]]
                                if gu > gv:
                                    gu, gv = gv, gu
                                q3_edges_found.append((gu, gv))
                            break
                    if found_in_subset:
                        break
                if found_in_subset:
                    break
            
            if found_in_subset:
                found_q3 = True
                best_edge = None
                max_deg_sum = -1
                for e in q3_edges_found:
                    u, v = e
                    deg_sum = degrees[u] + degrees[v]
                    if deg_sum > max_deg_sum:
                        max_deg_sum = deg_sum
                        best_edge = e
                
                u, v = best_edge
                edges.remove(best_edge)
                adj[u][v] = False
                adj[v][u] = False
                degrees[u] -= 1
                degrees[v] -= 1
                
                for sub_idx in edge_to_subsets[best_edge]:
                    subset_edge_counts[sub_idx] -= 1
                    
                break
        
        if not found_q3:
            break
            
    return [[u, v] for u, v in edges]
