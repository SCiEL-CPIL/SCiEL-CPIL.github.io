import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate Clebsch Graph
    # Vertices 0..15 are 5-bit even parity strings
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    clebsch_edges = []
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges) # 40
    
    # 2. Random Search
    # Pre-compute all pairs
    all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
    
    def is_3_colorable(adj_matrix):
        n = 16
        colors = [-1] * n
        
        def backtrack(v):
            if v == n:
                return True
            # Heuristic: try colors
            for c in range(3):
                valid = True
                for u in range(n):
                    if adj_matrix[v][u] and colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    while time.time() < deadline:
        random.shuffle(all_pairs)
        adj = [[0]*16 for _ in range(16)]
        edges = []
        
        for u, v in all_pairs:
            # Check if adding (u,v) creates a triangle
            ok = True
            for w in range(16):
                if adj[u][w] and adj[v][w]:
                    ok = False
                    break
            if ok:
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
                
        if len(edges) > best_count:
            # Check if not 3-colorable
            if not is_3_colorable(adj):
                best_edges = edges
                best_count = len(edges)
                
    return best_edges
