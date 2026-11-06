"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    best_edges = None
    best_score = float('inf')
    
    # Try multiple partition sizes: K_{a, 16-a} + (65 - a*(16-a)) internal edges
    # a=6: 60 edges, need 5 internal
    # a=7: 63 edges, need 2 internal
    # a=8: 64 edges, need 1 internal
    # a=5: 55 edges, need 10 internal
    
    for small_size in [5, 6, 7, 8]:
        if time.time() > deadline - 15.0:
            break
        
        large_size = n - small_size
        base_edges = [[u, v] for u in range(small_size) for v in range(small_size, n)]
        num_internal = target_edges - len(base_edges)
        
        if num_internal < 0 or num_internal > large_size * (large_size - 1) // 2:
            continue
        
        large_vertices = list(range(small_size, n))
        all_internal_edges = [(a, b) for a in range(small_size, n) for b in range(a+1, n)]
        
        # Generate multiple random initial configurations
        num_restarts = 20 if small_size == 6 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Random selection of internal edges
            random.shuffle(all_internal_edges)
            random_internal = all_internal_edges[:num_internal]
            
            current_edges = base_edges + [[a, b] for a, b in random_internal]
            
            if not is_valid(current_edges):
                continue
            
            current_score = compute_score(current_edges)
            
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges[:]
            
            # Local search with diverse moves
            if time.time() > deadline - 10.0:
                continue
            
            # Local search: try edge swaps in the large part
            adj, deg = build_adj(current_edges)
            current_score = compute_score_from_adj(adj, deg)
            
            internal_set = set()
            for u, v in current_edges:
                if u >= small_size and v >= small_size:
                    internal_set.add((min(u,v), max(u,v)))
            
            improved = True
            max_ls_iters = 20000 if small_size == 6 else 10000
            iters = 0
            while improved and iters < max_ls_iters:
                improved = False
                iters += 1
                if time.time() > deadline - 5.0:
                    break
                
                internal_list = list(internal_set)
                random.shuffle(internal_list)
                
                for u1, v1 in internal_list:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Try several random replacements
                    for _ in range(5):
                        u2 = random.randint(small_size, n-1)
                        v2 = random.randint(small_size, n-1)
                        if u2 == v2:
                            continue
                        u2, v2 = min(u2, v2), max(u2, v2)
                        if (u2, v2) == (u1, v1) or (u2, v2) in internal_set:
                            continue
                        if (adj[u2] >> v2) & 1:
                            continue
                        
                        new_adj = adj[:]
                        new_deg = deg[:]
                        
                        new_adj[u1] &= ~(1 << v1)
                        new_adj[v1] &= ~(1 << u1)
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        
                        new_adj[u2] |= (1 << v2)
                        new_adj[v2] |= (1 << u2)
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score = compute_score_from_adj(new_adj, new_deg)
                        
                        if new_score < current_score:
                            adj = new_adj
                            deg = new_deg
                            current_score = new_score
                            internal_set.remove((u1, v1))
                            internal_set.add((u2, v2))
                            
                            # Rebuild edge list
                            current_edges = []
                            for u, v in current_edges:
                                if u >= small_size and v >= small_size:
                                    if (min(u,v), max(u,v)) != (u1, v1):
                                        current_edges.append([u, v])
                                else:
                                    current_edges.append([u, v])
                            current_edges.append([u2, v2])
                            
                            improved = True
                            break
                    if improved:
                        break
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = current_edges[:]
    
    # Final verification
    if not is_valid(best_edges):
        base_edges = [[u, v] for u in range(6) for v in range(6, 16)]
        best_edges = base_edges + [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
