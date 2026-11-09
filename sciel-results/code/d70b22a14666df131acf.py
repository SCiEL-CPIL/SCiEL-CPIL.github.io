import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    q = 5
    n = q*q + q + 1  # 31
    
    points = []
    for a in range(q):
        for b in range(q):
            points.append((1, a, b))
    for a in range(q):
        points.append((0, 1, a))
    points.append((0, 0, 1))
    
    def B(P, Q):
        return (P[0]*Q[0] + P[1]*Q[1] + P[2]*Q[2]) % q
    
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if B(points[i], points[j]) == 0:
                adj[i][j] = adj[j][i] = True
    
    triu_idx = np.triu_indices(n, 1)
    
    def count_c4(adj_mat):
        A = np.array(adj_mat, dtype=np.int32)
        A2 = A @ A
        vals = A2[triu_idx]
        c4_per_pair = np.where(vals >= 2, vals*(vals-1)//2, 0)
        return int(c4_per_pair.sum()) // 2
    
    assert count_c4(adj) == 0
    
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    
    def c4s_created(u, v, adj_mat):
        w_list = [w for w in range(n) if w != u and w != v and adj_mat[u][w]]
        x_list = [x for x in range(n) if x != u and x != v and adj_mat[v][x]]
        count = 0
        for w in w_list:
            for x in x_list:
                if w != x and adj_mat[w][x]:
                    count += 1
        return count
    
    def greedy_add():
        cur = [row[:] for row in adj]
        added = []
        for _ in range(3):
            best_e = None
            best_c = float('inf')
            for (u,v) in non_edges:
                if (u,v) in added:
                    continue
                c = c4s_created(u, v, cur)
                if c < best_c:
                    best_c = c
                    best_e = (u,v)
            if best_e:
                u,v = best_e
                cur[u][v] = cur[v][u] = True
                added.append(best_e)
        return cur
    
    def random_greedy():
        cur = [row[:] for row in adj]
        pair = random.sample(non_edges, 2)
        for (u,v) in pair:
            cur[u][v] = cur[v][u] = True
        best_e = None
        best_c = float('inf')
        used = set(pair)
        for (u,v) in non_edges:
            if (u,v) in used:
                continue
            c = c4s_created(u, v, cur)
            if c < best_c:
                best_c = c
                best_e = (u,v)
        if best_e:
            u,v = best_e
            cur[u][v] = cur[v][u] = True
        return cur
    
    candidates = []
    candidates.append(greedy_add())
    
    for _ in range(30):
        if time.time() > deadline - 60:
            break
        candidates.append(random_greedy())
    
    best_adj = None
    best_c4 = float('inf')
    for cand in candidates:
        c4 = count_c4(cand)
        if c4 < best_c4:
            best_c4 = c4
            best_adj = [row[:] for row in cand]
    
    def local_search(adj_mat, max_time):
        best_c4 = count_c4(adj_mat)
        
        def get_non_edges(a):
            return [(i,j) for i in range(n) for j in range(i+1,n) if not a[i][j]]
        
        non_e = get_non_edges(adj_mat)
        
        while time.time() < max_time:
            improved = False
            edge_list = [(i,j) for i in range(n) for j in range(i+1,n) if adj_mat[i][j]]
            random.shuffle(edge_list)
            for (ri, rj) in edge_list[:40]:
                if time.time() > max_time:
                    break
                adj_mat[ri][rj] = adj_mat[rj][ri] = False
                ne_sample = random.sample(non_e, min(60, len(non_e)))
                for (ai, aj) in ne_sample:
                    if (ai,aj) == (ri,rj) or (ai,aj) == (rj,ri):
                        continue
                    adj_mat[ai][aj] = adj_mat[aj][ai] = True
                    new_c4 = count_c4(adj_mat)
                    adj_mat[ai][aj] = adj_mat[aj][ai] = False
                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        adj_mat[ri][rj] = adj_mat[rj][ri] = False
                        adj_mat[ai][aj] = adj_mat[aj][ai] = True
                        non_e = get_non_edges(adj_mat)
                        improved = True
                        break
                if time.time() > max_time:
                    break
                if not improved:
                    adj_mat[ri][rj] = adj_mat[rj][ri] = True
            if not improved:
                break
        
        return adj_mat, best_c4
    
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        best_adj, best_c4 = local_search(best_adj, time.time() + remaining)
    
    final_edges = [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i][j]]
    if len(final_edges) != 91:
        # Fallback: polarity graph + 3 arbitrary non-edges
        fallback = [row[:] for row in adj]
        for (u,v) in non_edges[:3]:
            fallback[u][v] = fallback[v][u] = True
        final_edges = [[i, j] for i in range(n) for j in range(i+1, n) if fallback[i][j]]
    
    assert len(final_edges) == 91
    return final_edges
