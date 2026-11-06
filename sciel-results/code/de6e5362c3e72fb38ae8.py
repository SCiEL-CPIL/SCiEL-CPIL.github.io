import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    best_n = 39
    
    def is_c4_free(adj, n):
        # Check if graph is C4-free
        # A graph has a C4 if there exist two vertices u, v with at least 2 common neighbors
        # We can check this by looking at common neighbors for all pairs
        for i in range(n):
            for j in range(i+1, n):
                common = 0
                for k in range(n):
                    if adj[i][k] and adj[j][k]:
                        common += 1
                        if common >= 2:
                            return False
        return True
    
    def get_paley_graph(n):
        # Paley graph on n vertices where n is a prime power
        # For n=47, we need a prime. 47 is prime.
        # But Paley graph requires q ≡ 1 (mod 4). 47 ≡ 3 (mod 4), so not a Paley graph.
        # Let's try n=43 (prime, 43 ≡ 3 mod 4) or n=53 (too big)
        # Actually, the problem says n <= 47.
        # Let's just use a random graph approach but with a smarter seed if Paley doesn't work directly.
        # Or we can construct a specific C4-free graph.
        
        # For now, let's return None and use random graphs.
        return None
    
    def try_construct(n, seed):
        random.seed(seed)
        target_deg = n - 39
        
        # Try different densities
        densities = [0.25, 0.22, 0.20, 0.18, 0.16, 0.14]
        
        for p in densities:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Initial check
            degs = adj.sum(axis=1)
            if np.min(degs) < target_deg:
                continue
            
            # Greedy removal using set-based C4 repair for speed
            # Maintain a set of C4 violations
            # Instead of full matrix mult each time, we update incrementally
            
            for step in range(3000):
                if time.time() > deadline:
                    return None
                
                # Compute common neighbors matrix
                common = adj @ adj
                np.fill_diagonal(common, 0)
                
                # Check if valid
                if common.max() < 2:
                    if np.min(adj.sum(axis=1)) >= target_deg:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        break
                
                # Find violations
                indices = np.triu_indices(n, k=1)
                vals = common[indices]
                viol_mask = vals >= 2
                if not np.any(viol_mask):
                    break
                viol_positions = np.where(viol_mask)[0]
                pos = random.choice(viol_positions)
                u, v = indices[0][pos], indices[1][pos]
                
                # Find common neighbors
                commons = np.where((adj[u] & adj[v]) == 1)[0]
                
                if len(commons) == 0:
                    continue
                
                w = random.choice(commons.tolist())
                
                deg_u = adj[u].sum()
                deg_v = adj[v].sum()
                
                if deg_u > deg_v:
                    x = u
                elif deg_v > deg_u:
                    x = v
                else:
                    x = u if random.random() < 0.5 else v
                
                # Remove (x, w)
                adj[x][w] = 0
                adj[w][x] = 0
                
                # Check min degree
                if np.min(adj.sum(axis=1)) < target_deg:
                    break
        
        return None
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        for seed in range(300):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_result = res
                found = True
                if n == 47:
                    return best_result
                break
        if found:
            continue
        else:
            pass
    
    return best_result
