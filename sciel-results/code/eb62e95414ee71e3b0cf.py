"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 150s kill
    random.seed(42)
    
    def find_k4(adj, n):
        """Find one K4 in graph. Returns tuple of 4 vertices or None."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    common = []
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            common.append(k)
                    for a in range(len(common)):
                        for b in range(a+1, len(common)):
                            if adj[common[a]][common[b]]:
                                return (i, j, common[a], common[b])
        return None

    def find_indep6(adj, n):
        """Find one independent 6-set using backtracking."""
        # Build non-adjacency (independent) structure
        # A vertex v is "independent-compatible" with u if not adj[v][u]
        
        def backtrack(remaining, chosen):
            if len(chosen) == 6:
                return tuple(chosen)
            if len(remaining) < 6 - len(chosen):
                return None
            for idx, v in enumerate(remaining):
                # Try including v
                new_chosen = chosen + [v]
                # New remaining: vertices after idx that are non-adjacent to v
                new_remaining = []
                for u in remaining[idx+1:]:
                    if not adj[v][u]:
                        new_remaining.append(u)
                result = backtrack(new_remaining, new_chosen)
                if result is not None:
                    return result
            return None
        
        return backtrack(list(range(n)), [])

    def is_valid(adj, n):
        return find_k4(adj, n) is None and find_indep6(adj, n) is None
    
    def make_adj(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        return adj
    
    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    # ---- Construction 1: Paley graph on 17 vertices (known valid) ----
    squares_17 = {x*x % 17 for x in range(1, 17)}
    paley_edges_17 = [[i, j] for i in range(17) for j in range(i+1, 17) if (j-i) % 17 in squares_17]
    best_result = {'n': 17, 'edges': paley_edges_17}
    
    # ---- Construction 2: Product graph on 35 vertices (Exoo-style) ----
    # Try the known 35-vertex construction
    n35 = 35
    D5 = {1, 4}  # QR mod 5
    D7 = {1, 2, 4}  # QR mod 7
    
    adj35 = [[False]*n35 for _ in range(n35)]
    for i in range(n35):
        a1, b1 = divmod(i, 7)
        for j in range(i+1, n35):
            a2, b2 = divmod(j, 7)
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i][j] = adj35[j][i] = True
    
    if is_valid(adj35, n35):
        best_result = {'n': n35, 'edges': extract_edges(adj35, n35)}
        return best_result
    
    # ---- Construction 3: Try to repair the 35-vertex graph ----
    # If the product graph has violations, try to repair it
    adj_repair = [row[:] for row in adj35]
    
    for step in range(2000):
        if time.time() > deadline - 5:
            break
        
        k4 = find_k4(adj_repair, n35)
        if k4:
            i, j, k, l = k4
            pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
            u, v = random.choice(pair_options)
            adj_repair[u][v] = adj_repair[v][u] = False
            continue
        
        indep6 = find_indep6(adj_repair, n35)
        if indep6:
            u, v = random.sample(indep6, 2)
            adj_repair[u][v] = adj_repair[v][u] = True
            continue
        
        # Valid!
        if is_valid(adj_repair, n35):
            best_result = {'n': n35, 'edges': extract_edges(adj_repair, n35)}
            return best_result
    
    # ---- Construction 4: Local search from random starts for n=20..34 ----
    for target_n in range(34, 18, -1):
        if time.time() > deadline - 5:
            break
        
        best_adj = None
        
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
            
            # Start with random graph with appropriate edge density
            edge_prob = 0.45
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if random.random() < edge_prob:
                        edges.append([i, j])
            
            adj = make_adj(target_n, edges)
            
            # Repair loop
            for step in range(800):
                if time.time() > deadline - 5:
                    break
                
                # Check for K4
                k4 = find_k4(adj, target_n)
                if k4:
                    i, j, k, l = k4
                    pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                    u, v = random.choice(pair_options)
                    adj[u][v] = adj[v][u] = False
                    continue
                
                # Check for Independent Set of size 6
                indep6 = find_indep6(adj, target_n)
                if indep6:
                    # Add an edge within the independent set
                    # Prefer adding edges that break the most potential independent sets
                    u, v = random.sample(indep6, 2)
                    adj[u][v] = adj[v][u] = True
                    continue
                
                # If no violations found, graph is valid
                best_adj = [row[:] for row in adj]
                break
            
            if best_adj is not None:
                break
        
        if best_adj is not None:
            edges = extract_edges(best_adj, target_n)
            final_adj = make_adj(target_n, edges)
            if is_valid(final_adj, target_n):
                if best_result is None or target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
