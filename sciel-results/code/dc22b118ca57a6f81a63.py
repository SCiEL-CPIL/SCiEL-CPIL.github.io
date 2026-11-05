"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Try many random seeds for greedy construction
    # Greedy is fast, so we can run many iterations
    seeds = list(range(500))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        candidate = _greedy_construct(n, seed=seed)
        if len(candidate) > best_score:
            best_score = len(candidate)
            best_edges = candidate
            
    return best_edges

def _greedy_construct(n, seed=42):
    """
    Greedy construction of K2,3-free graph.
    """
    random.seed(seed)
    
    # common_neighbors[u][v] = number of common neighbors of u and v
    common = [[0] * n for _ in range(n)]
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    # Pre-generate all possible edges
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(possible_edges)
    
    for u, v in possible_edges:
        if adj[u][v]:
            continue
            
        # Check validity of adding (u, v)
        # 1. Direct check for (u, v)
        if common[u][v] >= 3:
            continue
            
        valid = True
        
        # 2. Check impact on pairs (a, v) where a is neighbor of u
        #    common[a][v] will increase by 1. Must stay < 3.
        for a in range(n):
            if adj[u][a] and a != v:
                if common[a][v] >= 2:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # 3. Check impact on pairs (u, b) where b is neighbor of v
        #    common[u][b] will increase by 1. Must stay < 3.
        for b in range(n):
            if adj[v][b] and b != u:
                if common[u][b] >= 2:
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Add edge (u, v)
        edges.append([u, v])
        adj[u][v] = adj[v][u] = True
        
        # Update common neighbor counts
        # For each neighbor a of u (a != v), common[a][v] increases by 1
        for a in range(n):
            if adj[u][a] and a != v:
                common[a][v] += 1
                common[v][a] += 1
                
        # For each neighbor b of v (b != u), common[u][b] increases by 1
        for b in range(n):
            if adj[v][b] and b != u:
                common[u][b] += 1
                common[b][u] += 1
                
    return edges
# EVOLVE-BLOCK-END
