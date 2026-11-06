import random
import time
import sys

def construct():
    """
    Constructs a K4-free graph with no independent set of size 6.
    Uses a randomized local search with optimized K4 counting and exact Alpha checking.
    """
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    # Helper: Count K4s
    def count_k4s(n, adj):
        k4s = 0
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]: continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            k4s += 1
        return k4s

    # Helper: Check if graph has independent set of size 6
    def has_alpha6(n, adj):
        # Backtracking search for independent set of size 6
        def dfs(current_set, candidates):
            if len(current_set) == 6:
                return True
            if len(candidates) < 6 - len(current_set):
                return False
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Include v
            new_candidates = [u for u in rest if not adj[v][u]]
            if dfs(current_set + [v], new_candidates):
                return True
            
            # Option 2: Exclude v
            if dfs(current_set, rest):
                return True
            
            return False
        
        return dfs([], list(range(n)))

    # Main Search Loop
    best_n = 17
    best_edges = []
    
    # Seed graph (Paley 17)
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    # Try to find larger graphs
    for n in range(18, 36):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(50):
            if time.time() - start_time > timeout:
                break
                
            # Initialize random graph
            adj = [[False]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Local Search
            current_k4s = count_k4s(n, adj)
            best_k4s = current_k4s
            best_adj = [row[:] for row in adj]
            
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                    
                # Pick random edge to flip
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j: continue
                if i > j: i, j = j, i
                
                old_val = adj[i][j]
                adj[i][j] = not old_val
                adj[j][i] = not old_val
                
                new_k4s = count_k4s(n, adj)
                
                if new_k4s < best_k4s:
                    best_k4s = new_k4s
                    best_adj = [row[:] for row in adj]
                    if best_k4s == 0:
                        # Check Alpha
                        if not has_alpha6(n, best_adj):
                            # Valid graph found
                            edges = []
                            for u in range(n):
                                for v in range(u+1, n):
                                    if best_adj[u][v]:
                                        edges.append([u, v])
                            best_n = n
                            best_edges = edges
                            found = True
                            break
                else:
                    # Revert
                    adj[i][j] = old_val
                    adj[j][i] = old_val
                    
            if found:
                break
                
        if found:
            # We found a valid graph for this n. 
            # Can we go higher? Yes, continue loop.
            pass

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
