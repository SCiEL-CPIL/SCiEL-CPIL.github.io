import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115
    
    n = 16
    
    def find_q3_edges(adj):
        # Randomize search order to find different Q3s
        vertices = list(range(n))
        random.shuffle(vertices)
        
        for v0 in vertices:
            neigh_v0 = adj[v0]
            neigh_list = []
            mask = neigh_v0
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if v != v0:
                    neigh_list.append(v)
            
            random.shuffle(neigh_list)
            
            for v1, v2, v4 in combinations(neigh_list, 3):
                common_12 = adj[v1] & adj[v2]
                common_14 = adj[v1] & adj[v4]
                common_24 = adj[v2] & adj[v4]
                
                mask3 = common_12
                while mask3:
                    v3 = (mask3 & -mask3).bit_length() - 1
                    mask3 &= mask3 - 1
                    if v3 == v0 or v3 == v1 or v3 == v2 or v3 == v4: continue
                    
                    mask5 = common_14
                    while mask5:
                        v5 = (mask5 & -mask5).bit_length() - 1
                        mask5 &= mask5 - 1
                        if v5 == v0 or v5 == v1 or v5 == v2 or v5 == v4 or v5 == v3: continue
                        
                        mask6 = common_24
                        while mask6:
                            v6 = (mask6 & -mask6).bit_length() - 1
                            mask6 &= mask6 - 1
                            if v6 == v0 or v6 == v1 or v6 == v2 or v6 == v4 or v6 == v3 or v6 == v5: continue
                            
                            common_356 = adj[v3] & adj[v5] & adj[v6]
                            mask7 = common_356
                            while mask7:
                                v7 = (mask7 & -mask7).bit_length() - 1
                                mask7 &= mask7 - 1
                                if v7 == v0 or v7 == v1 or v7 == v2 or v7 == v3 or v7 == v4 or v7 == v5 or v7 == v6: continue
                                
                                edges = [
                                    (min(v0,v1), max(v0,v1)),
                                    (min(v0,v2), max(v0,v2)),
                                    (min(v0,v4), max(v0,v4)),
                                    (min(v1,v3), max(v1,v3)),
                                    (min(v1,v5), max(v1,v5)),
                                    (min(v2,v3), max(v2,v3)),
                                    (min(v2,v6), max(v2,v6)),
                                    (min(v3,v7), max(v3,v7)),
                                    (min(v4,v5), max(v4,v5)),
                                    (min(v4,v6), max(v4,v6)),
                                    (min(v5,v7), max(v5,v7)),
                                    (min(v6,v7), max(v6,v7))
                                ]
                                return edges
        return None

    def choose_edge_to_remove(adj, q3_edges):
        # Heuristic: Remove edge with highest number of common neighbors
        scores = []
        for u, v in q3_edges:
            common = adj[u] & adj[v]
            cnt = bin(common).count('1')
            scores.append(cnt)
        
        max_score = max(scores)
        candidates = [q3_edges[i] for i in range(len(q3_edges)) if scores[i] == max_score]
        return random.choice(candidates)

    best_edges = 0
    best_adj = None
    
    for restart in range(30000):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(42 + restart)
        
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        while True:
            if time.time() - start_time > time_limit:
                break
                
            q3_edges = find_q3_edges(adj)
            if q3_edges is None:
                edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = adj[:]
                break
            else:
                u, v = choose_edge_to_remove(adj, q3_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
    if best_adj is not None:
        for it in range(50000):
            if time.time() - start_time > time_limit:
                break
            u = random.randint(0, 15)
            v = random.randint(0, 15)
            if u == v:
                continue
            if not ((best_adj[u] >> v) & 1):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                if find_q3_edges(best_adj) is None:
                    ec = sum(bin(best_adj[i]).count('1') for i in range(n)) // 2
                    if ec > best_edges:
                        best_edges = ec
                else:
                    best_adj[u] &= ~(1 << v)
                    best_adj[v] &= ~(1 << u)
    
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
