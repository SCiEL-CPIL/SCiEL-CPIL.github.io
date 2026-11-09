import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    q = 5
    n = 31
    
    # 1. Generate points of PG(2,5) efficiently
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                vec = [x, y, z]
                first_non_zero_idx = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_non_zero_idx = i
                        break
                if first_non_zero_idx == -1:
                    continue
                inv = pow(vec[first_non_zero_idx], q - 2, q)
                vec = [(v * inv) % q for v in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    if len(points) != 31:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    points_arr = np.array(points, dtype=int)
    
    # 2. Define Polarity Matrix A
    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=int)
    
    # 3. Compute adjacency using vectorized operations
    # adj[i,j] = 1 if points[i] @ A @ points[j] == 0 mod 5 and i != j
    # Compute all dot products at once
    # B = points @ A, shape (31, 3)
    B = (points_arr @ A) % q  # shape (31, 3)
    # C = B @ points.T, shape (31, 31)
    C = (B @ points_arr.T) % q
    adj = (C == 0).astype(int)
    np.fill_diagonal(adj, 0)
    
    # 4. Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    if len(edges) != 90:
        # Fallback
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # Build adjacency list
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    # Count C4s in base graph (should be 0 for polarity graph)
    def count_c4s(adj_mat, adj_lst):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = 0
                ni = adj_lst[i]
                nj = adj_lst[j]
                # Use set intersection for speed
                if len(ni) < len(nj):
                    si = set(ni)
                    for v in nj:
                        if v in si:
                            common += 1
                else:
                    sj = set(nj)
                    for v in ni:
                        if v in sj:
                            common += 1
                count += common
        return count // 2  # Each C4 counted twice (once for each diagonal pair)
    
    base_c4s = count_c4s(adj, adj_list)
    
    # 5. Find best edge to add greedily first
    best_edge = None
    min_cycles = float('inf')
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                continue
            # Count paths of length 3 from u to v
            count = 0
            for a in adj_list[u]:
                for b in adj_list[v]:
                    if adj[a, b] == 1:
                        count += 1
            if count < min_cycles:
                min_cycles = count
                best_edge = [u, v]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_edge is None:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    edges.append(best_edge)
    adj[best_edge[0]][best_edge[1]] = 1
    adj[best_edge[1]][best_edge[0]] = 1
    adj_list[best_edge[0]].append(best_edge[1])
    adj_list[best_edge[1]].append(best_edge[0])
    
    current_c4s = base_c4s + min_cycles
    
    # 6. Local search: try swaps (remove edge e1, add non-edge e2)
    # Delta in C4 count = C4s(new) - C4s(old)
    # C4s involving removed edge e1: subtract those
    # C4s involving added edge e2: add those
    # But C4s involving both e1 and e2 need careful handling
    
    def count_c4s_through_edge(adj_mat, u, v):
        """Count C4s that use edge (u,v)."""
        count = 0
        for a in adj_list[u]:
            if a == v:
                continue
            for b in adj_list[v]:
                if b == u:
                    continue
                if adj_mat[a][b] == 1:
                    count += 1
        return count
    
    # Build list of current edges and non-edges
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edge_set:
                non_edges.append((u, v))
    
    # Local search iterations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_swap = None
        
        for e1 in edges:
            u1, v1 = e1
            # C4s through e1
            c4s_e1 = count_c4s_through_edge(adj, u1, v1)
            
            for e2 in non_edges:
                u2, v2 = e2
                # C4s through e2 in current graph (before swap)
                # But we need to account for the fact that e1 is being removed
                # Approximate: count C4s through e2 ignoring e1
                c4s_e2 = 0
                for a in adj_list[u2]:
                    if a == v2:
                        continue
                    for b in adj_list[v2]:
                        if b == u2:
                            continue
                        if adj[a][b] == 1:
                            # Check if this C4 uses edge e1
                            # C4 is u2-a-b-v2-u2, uses edges (u2,a),(a,b),(b,v2),(v2,u2)
                            # It uses e1 if {u1,v1} is one of these edge pairs
                            if (min(a,b), max(a,b)) == (min(u1,v1), max(v1,v1)):
                                pass  # This C4 would be affected
                            elif (min(u2,a), max(u2,a)) == (min(u1,v1), max(u1,v1)):
                                pass
                            elif (min(b,v2), max(b,v2)) == (min(u1,v1), max(u1,v1)):
                                pass
                            else:
                                c4s_e2 += 1
                
                # More careful: count C4s through e2 that don't use e1
                # A C4 through e2=(u2,v2) uses path u2-x-y-v2 where x~u2, y~v2, x~y
                # It uses e1 if {x,y}={u1,v1} or {u2,x}={u1,v1} or {y,v2}={u1,v1}
                
                delta = c4s_e2 - c4s_e1
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if best_swap is not None and best_delta < 0:
            e1, e2 = best_swap
            # Perform swap
            edges.remove(e1)
            edge_set.remove((min(e1[0], e1[1]), max(e1[0], e1[1])))
            
            # Remove e1 from adjacency
            u1, v1 = e1
            adj[u1][v1] = 0
            adj[v1][u1] = 0
            adj_list[u1].remove(v1)
            adj_list[v1].remove(u1)
            
            # Add e2
            edges.append(e2)
            edge_set.add((min(e2[0], e2[1]), max(e2[0], e2[1])))
            u2, v2 = e2
            adj[u2][v2] = 1
            adj[v2][u2] = 1
            adj_list[u2].append(v2)
            adj_list[v2].append(u2)
            
            current_c4s += best_delta
            improved = True
    
    # Convert back to list format
    result = []
    for u, v in edges:
        result.append([u, v])
    
    return result
