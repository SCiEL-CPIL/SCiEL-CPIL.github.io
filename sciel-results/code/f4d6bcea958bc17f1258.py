import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Helper to convert bitmask adj to edge list
    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Helper to count K4s
    def count_k4(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = c
                        edges_from_w = adj[w] & rest
                        count += edges_from_w.bit_count()
        return count // 6

    # Helper to count Independent Sets of size 6 (K6 in complement)
    def count_ind6(n, adj):
        # Complement adjacency
        # comp_adj[u] has bits set for non-neighbors (excluding self)
        # Mask for valid vertices
        mask = (1 << n) - 1
        count = 0
        for u in range(n):
            # Non-neighbors of u
            # adj[u] has 1s for neighbors.
            # We want 1s for non-neighbors.
            # ~adj[u] & mask will have 1s for non-neighbors and self.
            # Clear self bit.
            comp_u = (~adj[u]) & mask & ~(1 << u)
            for v in range(u + 1, n):
                if (comp_u >> v) & 1: # u and v are NOT connected in G
                    # Common non-neighbors of u and v
                    # comp_v = (~adj[v]) & mask & ~(1 << v)
                    comp_v = (~adj[v]) & mask & ~(1 << v)
                    common_non = comp_u & comp_v
                    # We need to count edges in the complement graph within common_non?
                    # No, an independent set of size 6 in G is a K6 in G_bar.
                    # So we count K6s in G_bar.
                    # Logic is identical to count_k4 but on G_bar.
                    # However, implementing a separate counter is redundant.
                    # We can just call a generic count_clique_size(n, adj, size)
                    # But for speed, let's inline or reuse logic.
                    # For now, let's just use the same logic.
                    # Actually, let's write a generic function.
                    pass
        return 0 # Placeholder

    # Generic clique counter
    def count_cliques(n, adj, size):
        # This is getting complex. 
        # For size=4, we have the optimized version.
        # For size=6, the complexity is O(n^6) if naive.
        # But we can use the same "edge in common" logic?
        # A K6 is 6 vertices.
        # It's harder to count K6s efficiently with the same trick.
        # However, n is small (<=35).
        # Brute force K6 check:
        # Iterate over all 6-subsets? C(35,6) ~ 1.6M.
        # For each, check 15 edges. 24M ops.
        # In Python, this might be slow (~1-2s per check).
        # Too slow for SA.
        
        # Alternative:
        # We only need to check if *any* independent set of size 6 exists.
        # We don't need the count, just 0 or 1.
        # We can use a branch and bound or backtracking to find one.
        # Or, we can use the fact that if alpha(G) >= 6, we fail.
        
        # Let's try a different approach for the validity check:
        # 1. Check K4-free. (Fast)
        # 2. Check Alpha < 6.
        #    Use a greedy algorithm or simple backtracking to find an independent set of size 6.
        #    If found, invalid.
        
        def has_k4(n, adj):
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        c = common
                        while c:
                            lsb = c & -c
                            w = lsb.bit_length() - 1
                            c ^= lsb
                            rest = c
                            if (adj[w] & rest):
                                return True
            return False

        def has_ind6(n, adj):
            # Find independent set of size 6
            # Equivalent to finding K6 in complement.
            # Use backtracking on complement.
            # Or, use a simple recursive search for independent set in G.
            
            def find_is(current_set, candidates, size_needed):
                if len(current_set) == size_needed:
                    return True
                if len(candidates) < size_needed - len(current_set):
                    return False
                # Pick a vertex from candidates
                # Heuristic: pick vertex with lowest degree in induced subgraph of candidates?
                # For simplicity, just pick first.
                if not candidates:
                    return False
                
                # Try including the first candidate
                v = candidates[0]
                # New candidates must be non-neighbors of v in G
                new_candidates = [x for x in candidates[1:] if not ((adj[v] >> x) & 1)]
                if find_is(current_set + [v], new_candidates, size_needed):
                    return True
                
                # Try excluding v
                return find_is(current_set, candidates[1:], size_needed)
            
            return find_is([], list(range(n)), 6)

        # The backtracking might be slow if the graph is dense.
        # But for n=35, it's usually fast enough for a few checks.
        
        # SA Loop
        best_n = 17
        best_edges = None
        
        # Try to improve upon seed
        for target_n in [35, 30, 25, 20]:
            if time.time() > deadline - 10:
                break
                
            # Initialize random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Simulated Annealing
            temp = 1000.0
            cooling = 0.99
            current_score = 0 # 0 is valid
            
            # We want to minimize penalty: 1000 * has_k4 + 100 * has_ind6
            # Or just boolean flags.
            
            # Calculate initial penalty
            # This is expensive.
            
            # Let's just do hill climbing with restarts?
            # Or SA.
            
            # Given time constraints, a simple random restart with local search might be better.
            # Or, just try a few random graphs and check.
            # Probability of random graph being valid is low.
            
            # Let's stick to the seed if we can't find better quickly.
            # But the prompt implies we can improve.
            
            # Let's try to find a valid graph for n=20 first, as it's easier.
            # Then try 25, 30.
            
            pass # Logic to be filled in final code

    # Fallback to seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
