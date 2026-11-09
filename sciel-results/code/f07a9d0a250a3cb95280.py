import random
import time
import sys

def construct():
    n = 16
    target_edges = 65
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    # Initialize with seed graph: K_{8,8} + edge (0,1)
    # Vertices 0-7 and 8-15
    # Edges between all u in 0-7 and v in 8-15
    # Plus edge (0,1)
    
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    # Add K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add((u, v))
            
    # Add (0,1)
    adj[0][1] = 1
    adj[1][0] = 1
    edges.add((0, 1))
    
    # Helper to get score
    def get_score():
        deg = [0]*n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    deg[i] += 1
        
        max_sum = 0
        # Check all triples
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    best_score = get_score()
    best_adj = [row[:] for row in adj]
    best_edges = set(edges)
    
    current_score = best_score
    T = 10.0
    cooling = 0.999
    
    # Precompute list of all possible edges for random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Check time
        if time.time() > deadline:
            break
            
        # Generate move: remove one edge, add one edge
        # Pick random edge to remove
        if not edges:
            break
        rem_edge = random.choice(list(edges))
        
        # Pick random non-edge to add
        # We can pick randomly from all_pairs and check if it's in edges
        # Or maintain a list of non-edges. For n=16, 120 pairs total, 65 edges, 55 non-edges.
        # Picking randomly from all 120 and checking is fast enough.
        
        add_edge = None
        for _ in range(100): # Try up to 100 times to find a valid non-edge
            candidate = random.choice(all_pairs)
            if candidate not in edges:
                add_edge = candidate
                break
        if add_edge is None:
            continue
            
        if add_edge == rem_edge:
            continue
            
        # Apply move
        u1, v1 = rem_edge
        adj[u1][v1] = 0
        adj[v1][u1] = 0
        edges.remove(rem_edge)
        
        u2, v2 = add_edge
        adj[u2][v2] = 1
        adj[v2][u2] = 1
        edges.add(add_edge)
        
        new_score = get_score()
        
        if new_score <= current_score or random.random() < pow(2.718, -(new_score - current_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_edges = set(edges)
        else:
            # Revert
            adj[u1][v1] = 1
            adj[v1][u1] = 1
            edges.add(rem_edge)
            
            adj[u2][v2] = 0
            adj[v2][u2] = 0
            edges.remove(add_edge)
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return best found
    res_edges = []
    for u, v in best_edges:
        res_edges.append([u, v])
        
    return {'n': n, 'edges': res_edges}
