"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Enumerate PG(2,5) points in canonical form
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    canon = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    canon = (0, 1, (z * inv_y) % 5)
                else:
                    canon = (0, 0, 1)
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    assert len(points) == 31
    
    # Build polarity graph: edge if dot product = 0 mod 5
    def dot(p, q):
        return (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
    
    adj = np.zeros((31, 31), dtype=np.int8)
    for i in range(31):
        for j in range(i+1, 31):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = 1
                adj[j][i] = 1
    
    edges = [(i, j) for i in range(31) for j in range(i+1, 31) if adj[i][j]]
    
    # Adjust to exactly 91 edges
    edge_set = set(edges)
    random.seed(42)
    if len(edge_set) > 91:
        while len(edge_set) > 91:
            e = random.choice(list(edge_set))
            edge_set.remove(e)
    elif len(edge_set) < 91:
        candidates = [(i,j) for i in range(31) for j in range(i+1, 31) if (i,j) not in edge_set]
        random.shuffle(candidates)
        for e in candidates:
            if len(edge_set) >= 91:
                break
            edge_set.add(e)
    
    # Rebuild adjacency from adjusted edge set
    adj = np.zeros((31, 31), dtype=np.int8)
    for (u,v) in edge_set:
        adj[u][v] = 1
        adj[v][u] = 1
    
    # Count C4s using numpy matrix multiplication
    def count_c4s(adj_mat):
        A2 = adj_mat @ adj_mat
        total = 0
        for u in range(31):
            for v in range(u+1, 31):
                c = int(A2[u,v])
                total += c * (c - 1) // 2
        return total // 2
    
    best_c4 = count_c4s(adj)
    best_edge_set = set(edge_set)
    
    # Local search with edge swaps
    random.seed(123)
    for _ in range(300):
        if time.time() > deadline:
            break
        e_remove = random.choice(list(edge_set))
        non_edges = [(i,j) for i in range(31) for j in range(i+1, 31) if (i,j) not in edge_set]
        if not non_edges:
            break
        e_add = random.choice(non_edges)
        
        u1, v1 = e_remove
        u2, v2 = e_add
        
        # Apply swap
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        edge_set.discard(e_remove)
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        edge_set.add(e_add)
        
        new_c4 = count_c4s(adj)
        
        if new_c4 < best_c4:
            best_c4 = new_c4
            best_edge_set = set(edge_set)
        else:
            # Revert
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            edge_set.add(e_remove)
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            edge_set.discard(e_add)
    
    result = [[u, v] for (u, v) in best_edge_set]
    assert len(result) == 91
    return result
# EVOLVE-BLOCK-END
