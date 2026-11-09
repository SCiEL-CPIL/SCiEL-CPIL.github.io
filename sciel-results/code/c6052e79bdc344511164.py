"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin before 300s kill
    
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    
    best_n = n
    best_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    def get_k4s_and_indep_k4s(n, adj):
        """Precompute all K4s and independent K4s."""
        k4s = []
        indep_k4s = []
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            # Check if clique
            if adj[a][b] and adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d] and adj[c][d]:
                k4s.append(combo)
            # Check if independent
            if not adj[a][b] and not adj[a][c] and not adj[a][d] and not adj[b][c] and not adj[b][d] and not adj[c][d]:
                indep_k4s.append(combo)
        return k4s, indep_k4s
    
    def try_add_vertex(n, adj, k4s, indep_k4s, max_attempts=2000):
        """
        Try to add a new vertex by finding a valid neighbor set.
        Returns (success, new_n, new_adj, new_k4s, new_indep_k4s) or None.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly initialize neighbor set
            N = [random.random() < 0.5 for _ in range(n)]
            
            # Local search to fix conflicts
            for iteration in range(500):
                if time.time() > deadline:
                    return None
                
                # Check for violations
                viol_k4 = None
                viol_indep_k4 = None
                cost = 0
                
                for k4 in k4s:
                    if all(N[v] for v in k4):
                        cost += 1
                        if viol_k4 is None:
                            viol_k4 = k4
                
                for ik4 in indep_k4s:
                    if not any(N[v] for v in ik4):
                        cost += 1
                        if viol_indep_k4 is None:
                            viol_indep_k4 = ik4
                
                if cost == 0:
                    # Found valid neighbor set, add the vertex
                    new_n = n + 1
                    new_adj = [row[:] for row in adj]
                    new_adj.append([False] * new_n)
                    for i in range(n):
                        new_adj[new_n - 1][i] = N[i]
                        new_adj[i][new_n - 1] = N[i]
                    
                    # Recompute K4s and indep K4s for the new graph
                    new_k4s, new_indep_k4s = get_k4s_and_indep_k4s(new_n, new_adj)
                    return (True, new_n, new_adj, new_k4s, new_indep_k4s)
                
                # Fix one violation
                if viol_k4:
                    v = random.choice(viol_k4)
                    N[v] = False
                elif viol_indep_k4:
                    v = random.choice(viol_indep_k4)
                    N[v] = True
                else:
                    # Should not happen, but flip a random bit to escape
                    v = random.randint(0, n - 1)
                    N[v] = not N[v]
        
        return None
    
    # Precompute K4s and indep K4s for initial graph
    k4s, indep_k4s = get_k4s_and_indep_k4s(n, adj)
    
    # Try to add vertices up to 42
    target_max_n = 42
    
    while n < target_max_n:
        if time.time() > deadline:
            break
        
        result = try_add_vertex(n, adj, k4s, indep_k4s, max_attempts=1500)
        
        if result is None:
            break
        
        success, n, adj, k4s, indep_k4s = result
        best_n = n
        best_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
