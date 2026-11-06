import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 135  # seconds; leave safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    def deadline_exceeded():
        return elapsed() > DEADLINE
    
    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_triangle(n, adj):
        # For each edge (u,v), check if they share a common neighbor
        for u in range(n):
            au = adj[u]
            # Only consider neighbors v > u to avoid double counting
            v_mask = au & ~((1 << (u + 1)) - 1)
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                # Common neighbors of u and v
                common = au & adj[v]
                if common:
                    return True
        return False
    
    def max_independent_set_size(n, adj, limit):
        """
        Returns the size of the maximum independent set, 
        but stops early if it reaches 'limit'.
        Uses branch and bound with bitmasks.
        """
        # Sort vertices by degree in descending order to help pruning?
        # Actually, for bitmasks, we usually just iterate.
        # Let's use a recursive approach with memoization? No, state space too big.
        # Simple recursive B&B:
        # State: (candidates_mask, current_size)
        # We want to find max size.
        
        # Optimization: If candidates is empty, return 0.
        # If current_size + popcount(candidates) <= best_found, prune.
        
        best = [0]
        
        def dfs(candidates, size):
            if deadline_exceeded():
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            # Upper bound
            # We can use a greedy coloring bound on the complement graph (which is the original graph)
            # But computing coloring is expensive.
            # Simple bound: size + popcount(candidates)
            if size + bin(candidates).count('1') <= best[0]:
                return
            
            # Pick a vertex to branch on.
            # Heuristic: pick vertex with highest degree in the induced subgraph?
            # Or just the first one.
            # Let's pick the first available vertex.
            v = (candidates & -candidates).bit_length() - 1
            candidates_without_v = candidates & ~(1 << v)
            
            # Branch 1: Include v
            # New candidates: candidates_without_v & ~adj[v]
            new_cands_incl = candidates_without_v & ~adj[v]
            dfs(new_cands_incl, size + 1)
            
            # Branch 2: Exclude v
            dfs(candidates_without_v, size)
            
            # Early termination if we found a set of size >= limit
            if best[0] >= limit:
                return
        
        dfs((1 << n) - 1, 0)
        return best[0]
    
    def is_valid(n, edges):
        if deadline_exceeded():
            return False
        adj = build_adj(n, edges)
        if has_triangle(n, adj):
            return False
        # Check alpha < 10
        alpha = max_independent_set_size(n, adj, 10)
        if alpha >= 10:
            return False
        return True
    
    def andrasfai_9():
        """
        Construct the Andrasfai graph And(9) on 26 vertices.
        Vertices 0..25.
        Edges connect i and j if (j-i) mod 26 is in {1, 2, 4, 5, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25}?
        Actually, And(k) is a specific graph.
        For k=9, n=3k-1=26.
        It is a circulant graph.
        The connection set S is such that it's triangle free and alpha=3.
        For And(3) (n=8), S={1,2,4,5,7}? No, And(3) is C8 with chords.
        Let's use the known construction for And(9).
        Or simpler: use the known valid graph on 26 vertices.
        The seed program used C_n(1 mod 3) for n=26.
        Let's verify if C_26(1 mod 3) is valid.
        n=26, 26 % 3 = 2 != 0.
        Triangle free? Yes.
        Alpha? For C_n(1 mod 3), alpha is roughly n/3.
        For n=26, alpha is around 8 or 9.
        If alpha < 10, it's valid.
        The seed program returned this as a fallback and it scored 1.0 (n=26).
        So we can use this as a safe baseline.
        """
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    edges.append([i, j])
        return n, edges
    
    # Try to find a larger graph using random search + local improvement
    # Strategy:
    # 1. Start with a random triangle-free graph on n=39 (or 40).
    # 2. Check if alpha < 10.
    # 3. If not, try to add edges to reduce alpha, while maintaining triangle-free property.
    # 4. If stuck, restart.
    
    best_n, best_edges = andrasfai_9()
    if is_valid(best_n, best_edges):
        pass # Baseline established
    
    # Target n values: 39, 40
    # R(3,10) <= 41, so max n is 40.
    
    for target_n in [40, 39]:
        if deadline_exceeded():
            break
        
        # Attempt multiple restarts
        for restart in range(100):
            if deadline_exceeded():
                break
            
            # Generate random triangle-free graph
            # Start with empty graph
            edges = []
            adj = [0] * target_n
            
            # Add random edges
            # Probability of adding edge ~ 0.3
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < 0.3:
                        # Check if adding (u,v) creates a triangle
                        # Common neighbors of u and v
                        common = adj[u] & adj[v]
                        if not common:
                            edges.append([u, v])
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            
            # Now try to improve: add edges to reduce alpha
            # We want alpha < 10.
            # If alpha >= 10, find an independent set I of size 10.
            # Try to add edges within I that don't create triangles.
            
            improved = True
            while improved and not deadline_exceeded():
                improved = False
                # Check validity
                # We need to know if alpha < 10.
                # Computing exact alpha is expensive.
                # Let's use a heuristic: try to find an independent set of size 10.
                # If we can't find one quickly, maybe it doesn't exist?
                # Or we can just try to add edges randomly and check validity at the end.
                
                # Better approach:
                # Try to add a random edge (u,v) that doesn't create a triangle.
                # If we can add it, do so.
                # This tends to increase density and reduce alpha.
                
                # Pick random non-adjacent pair
                for _ in range(100): # Limit attempts per loop
                    u = random.randint(0, target_n - 1)
                    v = random.randint(0, target_n - 1)
                    if u == v or (adj[u] >> v) & 1:
                        continue
                    
                    common = adj[u] & adj[v]
                    if not common:
                        # Add edge
                        edges.append([u, v])
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
                
                # Occasionally check if we are valid
                if random.random() < 0.1:
                    if is_valid(target_n, edges):
                        best_n = target_n
                        best_edges = edges
                        return {'n': best_n, 'edges': best_edges}
            
            # Final check for this restart
            if is_valid(target_n, edges):
                best_n = target_n
                best_edges = edges
                return {'n': best_n, 'edges': best_edges}
        
        # If we reached here, we didn't find a valid graph for this target_n
        # Try smaller target_n?
    
    # Return best found
    return {'n': best_n, 'edges': best_edges}
