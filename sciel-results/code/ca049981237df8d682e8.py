import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    n = 31
    target_edges = 91
    q = 5
    
    # 1. Construct Polarity Graph of PG(2, 5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    assert len(points) == 31
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                edges.append((i, j))
                adj[i, j] = 1
                adj[j, i] = 1
    
    def count_c4_fast(adj_mat):
        """Fast C4 count using matrix multiplication."""
        A2 = np.dot(adj_mat, adj_mat)
        np.fill_diagonal(A2, 0)
        s2 = np.sum(A2 * A2)
        s1 = np.sum(A2)
        return (s2 - s1) // 4
    
    def delta_c4_remove(adj_mat, u, v):
        """Compute delta in C4 count when removing edge (u,v)."""
        # C4s containing edge (u,v) = common neighbors of u and v
        common = np.sum(adj_mat[u] & adj_mat[v])
        # Each C4 with edge (u,v) is counted once
        return -common
    
    def delta_c4_add(adj_mat, u, v):
        """Compute delta in C4 count when adding edge (u,v)."""
        # C4s that would be created by adding edge (u,v) = common neighbors of u and v
        common = np.sum(adj_mat[u] & adj_mat[v])
        return common
    
    current_edges = len(edges)
    
    # Adjust edge count to target_edges using incremental updates
    if current_edges > target_edges:
        to_remove = current_edges - target_edges
        for _ in range(to_remove):
            best_edge = None
            best_delta = float('inf')
            for e in edges:
                u, v = e
                delta = delta_c4_remove(adj, u, v)
                if delta < best_delta:
                    best_delta = delta
                    best_edge = e
            if best_edge:
                u, v = best_edge
                edges.remove(best_edge)
                adj[u, v] = 0
                adj[v, u] = 0
                
    elif current_edges < target_edges:
        to_add = target_edges - current_edges
        for _ in range(to_add):
            best_edge = None
            best_delta = float('inf')
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] == 0:
                        delta = delta_c4_add(adj, u, v)
                        if delta < best_delta:
                            best_delta = delta
                            best_edge = (u, v)
            if best_edge:
                u, v = best_edge
                edges.append(best_edge)
                adj[u, v] = 1
                adj[v, u] = 1

    # Now we have exactly 91 edges.
    # Perform Local Search with incremental C4 updates
    current_c4 = count_c4_fast(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    temp = 2.0
    cooling = 0.9995
    
    # Precompute list of non-edges for faster sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(edges)
    
    while time.time() < deadline:
        # Pick random edge to remove
        e_idx = random.randint(0, len(edges) - 1)
        u, v = edges[e_idx]
        
        # Pick random non-edge to add
        found = False
        attempts = 0
        while not found and attempts < 50:
            a, b = random.choice(all_pairs)
            if (a, b) not in edge_set and not (a == u and b == v):
                found = True
            attempts += 1
            
        if not found:
            continue
            
        # Compute deltas incrementally
        delta_remove = delta_c4_remove(adj, u, v)
        delta_add = delta_c4_add(adj, a, b)
        total_delta = delta_remove + delta_add
        
        new_c4 = current_c4 + total_delta
        
        if new_c4 < best_c4:
            # Apply move
            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1
            edges[e_idx] = (a, b)
            edge_set.remove((u, v))
            edge_set.add((a, b))
            current_c4 = new_c4
            best_c4 = new_c4
            best_adj = adj.copy()
        else:
            # Metropolis criterion
            if total_delta > 0 and random.random() < np.exp(-total_delta / temp):
                # Accept worse move
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                edges[e_idx] = (a, b)
                edge_set.remove((u, v))
                edge_set.add((a, b))
                current_c4 = new_c4
            # Else reject (no change)
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1

    # Reconstruct final edges from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
