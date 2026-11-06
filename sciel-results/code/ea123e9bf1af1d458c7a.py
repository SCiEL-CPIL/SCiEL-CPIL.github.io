"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices.
    Uses bitset-based local search for speed.
    """
    max_time = 115
    start_time = time.time()
    
    def time_check():
        return time.time() - start_time > max_time

    def find_independent_set(adj, n, target_size):
        """
        Finds an independent set of size target_size using DFS with pruning.
        Returns the list of vertices or None.
        """
        candidates = list(range(n))
        
        def dfs(current, cands):
            if len(current) + len(cands) < target_size:
                return None
            if len(current) == target_size:
                return current[:]
            
            # Heuristic: Pick vertex with min degree in induced subgraph
            min_deg = n
            min_v_idx = 0
            mask = 0
            for u in cands:
                mask |= 1 << u
                
            for idx, v in enumerate(cands):
                deg = bin(adj[v] & mask).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v_idx = idx
            
            v = cands[min_v_idx]
            
            # Include v
            new_cands = [u for u in cands if u != v and not (adj[v] >> u) & 1]
            res = dfs(current + [v], new_cands)
            if res: return res
            
            # Exclude v
            cands.pop(min_v_idx)
            res = dfs(current, cands)
            if res: return res
            cands.insert(min_v_idx, v)
            
            return None
        
        return dfs([], candidates)

    def solve(seed):
        rng = random.Random(seed)
        n = 40
        adj = [0] * n
        
        # Initialize with random edges (sparse to allow triangle removal)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        
        # Step 1: Make triangle-free
        changed = True
        while changed and not time_check():
            changed = False
            for i in range(n):
                if changed: break
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        if (adj[i] & adj[j]) != 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            break
        
        # Step 2: Reduce alpha by adding edges
        for _ in range(200):
            if time_check():
                return None
            
            indep_set = find_independent_set(adj, n, 10)
            
            if not indep_set:
                # No independent set of size 10 found
                return adj, n
            
            # Try to add an edge to break the independent set
            found_edge = False
            for i in range(len(indep_set)):
                for j in range(i + 1, len(indep_set)):
                    u = indep_set[i]
                    v = indep_set[j]
                    if not (adj[u] >> v) & 1:
                        if (adj[u] & adj[v]) == 0:
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
                            found_edge = True
                            break
                if found_edge:
                    break
            
            if not found_edge:
                # Stuck, break to try next seed
                break

        return None

    best_n = 0
    best_adj = None
    
    for seed in range(1000):
        if time_check():
            break
        
        res = solve(seed)
        if res:
            adj, n = res
            if n > best_n:
                best_n = n
                best_adj = adj
                if n >= 39:
                    break

    if best_adj is None:
        return {'n': 1, 'edges': []}
    
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
