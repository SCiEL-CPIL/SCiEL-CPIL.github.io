# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    n = 20
    
    def is_k23_free(adj):
        """Check if adjacency matrix has no K2,3 (no pair with 3+ common neighbors)."""
        common = adj @ adj
        for u in range(n):
            for v in range(u + 1, n):
                if common[u, v] >= 3:
                    return False
        return True
    
    def edge_count(adj):
        return int(np.sum(adj) // 2)
    
    def common_neighbors_count(adj, u, v):
        return int(np.sum(adj[u] * adj[v]))
    
    # Phase 1: Initialize with a good spectral construction
    # Use a Paley-type graph based on quadratic residues mod a prime, then adapt to n=20
    # For n=20, use a construction based on finite field properties
    
    # Strategy: Start with a random-ish but structured graph, then optimize
    # Use a greedy construction: add edges that don't create K2,3
    
    def greedy_construct():
        """Build a K2,3-free graph greedily, trying to maximize edges."""
        adj = np.zeros((n, n), dtype=int)
        # Try multiple random orderings and keep the best
        best_adj = None
        best_count = 0
        
        for trial in range(50):
            random.seed(trial * 12345 + 7)
            adj = np.zeros((n, n), dtype=int)
            
            # Generate edge candidates in random order
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding edge (u,v) creates a K2,3
                # A K2,3 is created if there exist vertices a,b such that
                # {u,v} and {a,b,c} form K2,3, i.e., a,b,c are common neighbors of u and v
                # But actually, adding edge (u,v) could create K2,3 in two ways:
                # 1. u and v now have 3+ common neighbors (the edge itself doesn't affect this directly)
                # 2. For some w, u and w now have 3+ common neighbors because v becomes a common neighbor
                # 3. For some w, v and w now have 3+ common neighbors because u becomes a common neighbor
                
                # Check: after adding (u,v), do u and v have 3+ common neighbors?
                # Common neighbors of u and v: vertices w where adj[u,w]=1 and adj[v,w]=1
                # (excluding u and v themselves, but adj[u,u]=0 so this is automatic)
                cn_uv = int(np.sum(adj[u] * adj[v]))
                if cn_uv >= 3:
                    continue
                
                # Check for each w != u,v: does u and w now have 3+ common neighbors?
                # Before adding, common neighbors of u and w = sum of adj[u,x]*adj[w,x]
                # After adding (u,v), v becomes a common neighbor of u and w if adj[w,v]=1
                # So new count = old count + (1 if adj[w,v]=1 else 0)
                # We need this to be < 3 for all w
                
                valid = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    old_cn = int(np.sum(adj[u] * adj[w]))
                    new_cn = old_cn + (1 if adj[w, v] == 1 else 0)
                    if new_cn >= 3:
                        valid = False
                        break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Similarly check v and w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    old_cn = int(np.sum(adj[v] * adj[w]))
                    new_cn = old_cn + (1 if adj[w, u] == 1 else 0)
                    if new_cn >= 3:
                        valid = False
                        break
                    if not valid:
                        break
                
                if valid:
                    adj[u, v] = 1
                    adj[v, u] = 1
            
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj.copy()
        
        return best_adj, best_count
    
    # Phase 2: Local optimization via edge swaps (simulated annealing)
    def local_optimize(adj, time_limit):
        """Improve the graph using edge additions and swaps."""
        adj = adj.copy()
        current_count = edge_count(adj)
        best_adj = adj.copy()
        best_count = current_count
        
        # Precompute common neighbor counts for efficiency
        # common[u,v] = number of common neighbors of u and v
        def compute_common(adj):
            return adj @ adj
        
        common = compute_common(adj)
        
        # Temperature for simulated annealing
        temp = 10.0
        cooling = 0.9995
        
        while time.time() - start_time < time_limit:
            # Try random edge addition
            u, v = random.sample(range(n), 2)
            if adj[u, v] == 0:
                # Check if we can add this edge
                cn = int(common[u, v])
                if cn < 3:
                    # Check impact on other pairs
                    valid = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        # u and w: v becomes common neighbor if adj[w,v]=1
                        if adj[w, v] == 1 and int(common[u, w]) + 1 >= 3:
                            valid = False
                            break
                        # v and w: u becomes common neighbor if adj[w,u]=1
                        if adj[w, u] == 1 and int(common[v, w]) + 1 >= 3:
                            valid = False
                            break
                    if valid:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        common = compute_common(adj)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj.copy()
                        temp = max(0.1, temp * cooling)
                        continue
            
            # Try edge removal and re-addition (swap)
            # Remove a random edge, then try to add a different one
            edges_to_remove = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
            if edges_to_remove:
                ru, rv = random.choice(edges_to_remove)
                adj[ru, rv] = 0
                adj[rv, ru] = 0
                common = compute_common(adj)
                current_count -= 1
                
                # Try to add a new edge
                for _ in range(10):
                    nu, nv = random.sample(range(n), 2)
                    if adj[nu, nv] == 0:
                        cn = int(common[nu, nv])
                        if cn < 3:
                            valid = True
                            for w in range(n):
                                if w == nu or w == nv:
                                    continue
                                if adj[w, nv] == 1 and int(common[nu, w]) + 1 >= 3:
                                    valid = False
                                    break
                                if adj[w, nu] == 1 and int(common[nv, w]) + 1 >= 3:
                                    valid = False
                                    break
                            if valid:
                                adj[nu, nv] = 1
                                adj[nv, nu] = 1
                                common = compute_common(adj)
                                current_count += 1
                                break
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj.copy()
                temp = max(0.1, temp * cooling)
        
        return best_adj, best_count
    
    # Run greedy construction
    adj, count = greedy_construct()
    
    # Run local optimization with remaining time
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        adj, count = local_optimize(adj, start_time + remaining - 2)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
