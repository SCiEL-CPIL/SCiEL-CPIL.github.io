"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 15
    deadline = time.time() + 80
    
    def is_valid(adj, degs):
        # For each edge (u,v), s in N(u)\{v}, t in N(v)\{u}, s!=t: deg(s)!=deg(t)
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for s in range(n):
                    if s == v or not adj[s][u]:
                        continue
                    ds = degs[s]
                    for t in range(n):
                        if t == u or t == s or not adj[v][t]:
                            continue
                        if ds == degs[t]:
                            return False
        return True
    
    # Start with K_{7,8}
    adj = [[0]*n for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = 1
            adj[j][i] = 1
    
    degs = [sum(adj[i]) for i in range(n)]
    assert is_valid(adj, degs)
    
    best_edges = 56
    best_adj = [row[:] for row in adj]
    
    candidate_edges = []
    for i in range(7):
        for j in range(i+1, 7):
            candidate_edges.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            candidate_edges.append((i, j))
    
    # Greedy add
    improved = True
    while improved:
        improved = False
        for (i, j) in candidate_edges:
            if time.time() > deadline:
                break
            if adj[i][j]:
                continue
            adj[i][j] = 1
            adj[j][i] = 1
            degs[i] += 1
            degs[j] += 1
            if is_valid(adj, degs):
                new_edges = sum(sum(row) for row in adj) // 2
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                    improved = True
            else:
                adj[i][j] = 0
                adj[j][i] = 0
                degs[i] -= 1
                degs[j] -= 1
        if time.time() > deadline:
            break
    
    # Swap: remove one edge, add another
    if time.time() < deadline - 20:
        adj = [row[:] for row in best_adj]
        degs = [sum(adj[i]) for i in range(n)]
        all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        
        for (ri, rj) in all_pairs:
            if time.time() > deadline - 5:
                break
            if not adj[ri][rj]:
                continue
            adj[ri][rj] = 0
            adj[rj][ri] = 0
            degs[ri] -= 1
            degs[rj] -= 1
            
            for (ai, aj) in all_pairs:
                if time.time() > deadline - 5:
                    break
                if adj[ai][aj]:
                    continue
                if (ai, aj) == (ri, rj):
                    continue
                adj[ai][aj] = 1
                adj[aj][ai] = 1
                degs[ai] += 1
                degs[aj] += 1
                if is_valid(adj, degs):
                    new_edges = sum(sum(row) for row in adj) // 2
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                adj[ai][aj] = 0
                adj[aj][ai] = 0
                degs[ai] -= 1
                degs[aj] -= 1
            
            adj[ri][rj] = 1
            adj[rj][ri] = 1
            degs[ri] += 1
            degs[rj] += 1
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    
    return edges
# EVOLVE-BLOCK-END
