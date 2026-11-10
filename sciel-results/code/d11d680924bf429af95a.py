"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n = 15
    deadline = time.time() + 85  # safety margin
    
    # Start with complete graph and remove edges to satisfy constraint
    # Constraint: no 3-edge path s-u-v-t with deg(s) == deg(t)
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def is_valid(adj):
        degs = get_degrees(adj)
        for u in range(n):
            for v in range(n):
                if u == v or not adj[u][v]:
                    continue
                # u-v is an edge, find s adjacent to u (s != v) and t adjacent to v (t != u)
                for s in range(n):
                    if s == v or not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == u or not adj[v][t]:
                            continue
                        if s == t:
                            continue
                        if degs[s] == degs[t]:
                            return False
        return True
    
    # Strategy: Start with K_{7,8} (56 edges) which is valid, then try to add edges
    # K_{7,8}: part A = {0..6}, part B = {7..14}
    adj = [[0]*n for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = 1
            adj[j][i] = 1
    
    # Verify K_{7,8} is valid
    assert is_valid(adj), "K_{7,8} should be valid"
    
    best_edges = sum(sum(row) for row in adj) // 2
    best_adj = [row[:] for row in adj]
    
    # Try to add edges within parts
    candidate_edges = []
    for i in range(7):
        for j in range(i+1, 7):
            candidate_edges.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidate_edges.append((i, j))
    
    # Try adding each candidate edge
    for (i, j) in candidate_edges:
        if time.time() > deadline:
            break
        adj[i][j] = 1
        adj[j][i] = 1
        if is_valid(adj):
            new_edges = sum(sum(row) for row in adj) // 2
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [row[:] for row in adj]
        else:
            adj[i][j] = 0
            adj[j][i] = 0
    
    # Try adding pairs of edges
    if time.time() < deadline - 10:
        # Reset to best
        adj = [row[:] for row in best_adj]
        
        # Try adding more edges on top of best
        for (i, j) in candidate_edges:
            if time.time() > deadline:
                break
            if adj[i][j]:
                continue
            adj[i][j] = 1
            adj[j][i] = 1
            if is_valid(adj):
                new_edges = sum(sum(row) for row in adj) // 2
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
            else:
                adj[i][j] = 0
                adj[j][i] = 0
    
    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    
    return edges
# EVOLVE-BLOCK-END
