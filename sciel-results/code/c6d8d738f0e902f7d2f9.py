import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 115  # Safety margin
    
    rng = random.Random(42)
    
    def count_c4(adj):
        A = adj.astype(np.int32)
        A2 = A @ A
        triu = np.triu(A2, 1)
        c4s = np.sum(triu * (triu - 1) // 2)
        return c4s // 2

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append([i, j])
        return edges

    def build_polarity_graph():
        pts = []
        for y in range(5):
            for z in range(5):
                pts.append((1, y, z))
        for z in range(5):
            pts.append((0, 1, z))
        pts.append((0, 0, 1))
        
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = pts[i]
                p2 = pts[j]
                if (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5 == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    best_adj = None
    best_c4 = float('inf')
    
    # Strategy 1: Start with Polarity Graph (93 edges) and remove 2
    # Strategy 2: Random restarts
    
    # Init 1: Polarity Graph
    adj_pol = build_polarity_graph()
    # Remove 2 random edges
    edges_pol = [(i, j) for i in range(n) for j in range(i + 1, n) if adj_pol[i, j] == 1]
    if len(edges_pol) >= 2:
        to_remove = rng.sample(edges_pol, 2)
        for u, v in to_remove:
            adj_pol[u, v] = 0
            adj_pol[v, u] = 0
            
        c4_pol = count_c4(adj_pol)
        if c4_pol < best_c4:
            best_c4 = c4_pol
            best_adj = adj_pol.copy()
            
        # Run SA from Polarity start
        T = 1.0
        cooling = 0.999
        current_c4 = c4_pol
        current_adj = adj_pol.copy()
        
        while time.time() < deadline - 20 and T > 0.01:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or current_adj[u, v] == 0:
                continue
            
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            if a == b or current_adj[a, b] == 1 or (a == u and b == v) or (a == v and b == u):
                continue
            
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[a, b] = 1
            current_adj[b, a] = 1
            
            new_c4 = count_c4(current_adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = current_adj.copy()
            else:
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_adj[a, b] = 0
                current_adj[b, a] = 0
                
            T *= cooling

    # Strategy 2: Random Restarts if time permits
    restarts = 0
    while time.time() < deadline - 10 and restarts < 10:
        restarts += 1
        adj = np.zeros((n, n), dtype=np.int8)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edges = rng.sample(all_pairs, m)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_c4 = count_c4(adj)
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
            
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 5 and T > 0.01:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or adj[u, v] == 0:
                continue
            
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            if a == b or adj[a, b] == 1 or (a == u and b == v) or (a == v and b == u):
                continue
            
            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1
            
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
            T *= cooling

    if best_adj is None:
        adj = np.zeros((n, n), dtype=np.int8)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edges = rng.sample(all_pairs, m)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        best_adj = adj

    return get_edges(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A = adj.astype(np.int32)
    A2 = A @ A
    triu = np.triu(A2, 1)
    c4s = np.sum(triu * (triu - 1) // 2) // 2
    print(f"C4: {c4s}")
