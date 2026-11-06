import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Known best: R(3, 10) > 39 is known (Schrijver). 
    # We aim for n=39 or n=40.
    # R(3, 10) <= 41, so max possible n is 40.
    
    best_n = 0
    best_edges = []
    
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in adj[i]:
                if j <= i: continue
                for k in adj[j]:
                    if k > j and k in adj[i]:
                        return False
        return True

    def has_independent_set(adj, n, k):
        """
        Exact check for independent set of size k using branch and bound.
        Returns True if found.
        """
        if k == 0:
            return True
        if n < k:
            return False
        
        # Upper bound: if we can't pick k more vertices from remaining, prune
        # Simple bound: remaining vertices count
        # Better bound: greedy coloring or clique cover of complement? 
        # For small n (<=40) and k=10, simple backtracking with pruning works.
        
        # Sort vertices by degree (ascending) to find independent set faster?
        # Actually, for independent set, we want vertices with high degree in G 
        # (low degree in complement) to constrain choices? 
        # No, standard independent set: pick v, then recurse on V \ ({v} U N(v)).
        
        # Heuristic: Pick vertex with maximum degree in the induced subgraph?
        # Or minimum degree? 
        # Let's use a simple recursive search with memoization or just backtracking.
        
        # To make it fast, we can use a greedy upper bound on the size of the 
        # maximum independent set in the remaining graph.
        # But for exact check, we just need to find ONE of size k.
        
        # We will use a recursive function with bitmasks for speed if n <= 64.
        # Here n <= 40, so bitmask fits in 64-bit int.
        
        # Precompute adjacency bitmasks
        # adj_mask[i] is a bitmask of neighbors of i
        
        # This function assumes adj is passed as list of sets for clarity, 
        # but for speed in the loop, we should convert to bitmasks outside.
        
        pass # Placeholder, we will implement the bitmask version below

    def check_valid_bitmask(adj_masks, n, k=10):
        """
        Returns True if graph with adj_masks on n vertices has no independent set of size k.
        """
        # We want to check if alpha(G) < k.
        # Equivalent to: Max Independent Set < k.
        
        # Use backtracking to find Independent Set of size k.
        # If found, return False (invalid).
        # If search space exhausted, return True (valid).
        
        # Heuristic for branching:
        # Pick a vertex v.
        # Branch 1: Include v. Remove v and its neighbors from candidate set.
        # Branch 2: Exclude v. Remove v from candidate set.
        
        # To speed up, we can use an upper bound on the remaining independent set size.
        # A simple upper bound is the number of remaining vertices.
        # A better one is greedy coloring of the complement? 
        # Or just the count. For n=40, k=10, simple backtracking might be slow if not pruned well.
        
        # Let's use a greedy upper bound: 
        # Greedy independent set size in the remaining graph. 
        # If greedy_size < k, prune.
        
        def greedy_indep_size(candidates_mask):
            count = 0
            rem = candidates_mask
            while rem:
                # Pick vertex with min degree in remaining? 
                # Or just first available?
                # For an upper bound, we want a *lower* bound on the size of the max independent set?
                # No, we need an upper bound on the size of the max independent set to prune.
                # Wait. If we want to find an independent set of size k, 
                # and we know the MAX independent set in the remaining graph is < k, we can prune.
                # Greedy gives a LOWER bound on the max independent set. 
                # So greedy_size <= alpha(remaining). 
                # This doesn't help prune "alpha < k". 
                
                # We need an UPPER bound on alpha(remaining).
                # Upper bound: n_remaining.
                # Better upper bound: Clique number of the complement graph (which is alpha of G).
                # Calculating clique number is also hard.
                
                # Alternative: Just use the count of remaining vertices.
                # This is weak.
                
                # Let's try a different pruning: 
                # If the number of remaining vertices < k, return 0 (can't form size k).
                
                # Actually, for n=40, k=10, the search tree is manageable if we order vertices well.
                # Let's stick to a simple DFS with bitmasks.
                
                # Pick the vertex with the highest degree in the induced subgraph?
                # High degree in G means low degree in Complement?
                # No, we are looking for Independent Set in G.
                # Independent Set in G = Clique in Complement.
                # So we are looking for a clique of size k in the Complement graph.
                # Clique finding is easier to prune with degree bounds?
                
                # Let's just do DFS for Independent Set in G.
                # State: candidates_mask, current_size
                # If current_size + popcount(candidates_mask) < k: return False (prune)
                # If current_size == k: return True
                
                # Branching:
                # Pick v from candidates.
                # 1. Include v: candidates = candidates & ~(1<<v) & ~adj_masks[v]
                # 2. Exclude v: candidates = candidates & ~(1<<v)
                
                # To improve pruning, pick v with high degree in the induced subgraph?
                # If v has high degree, including it removes many candidates, reducing the tree.
                # So pick v with max degree in the induced subgraph defined by candidates.
                
                pass # Implementation in the main loop for speed

        # Optimized check
        # We will use a recursive helper that returns True if an independent set of size k exists.
        
        def find_indep(candidates_mask, current_size):
            if current_size == k:
                return True
            
            # Pruning: not enough vertices left
            if (candidates_mask.bit_count() < k - current_size):
                return False
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in the induced subgraph
            # to maximize pruning in the "include" branch.
            best_v = -1
            max_deg = -1
            # Iterate through set bits
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Calculate degree of v in candidates
                # Neighbors of v in candidates: adj_masks[v] & candidates_mask
                deg = (adj_masks[v] & candidates_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                return False # No candidates left
            
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v
            # New candidates: remove best_v and its neighbors
            new_candidates = candidates_mask & ~v_bit & ~adj_masks[best_v]
            if find_indep(new_candidates, current_size + 1):
                return True
            
            # Branch 2: Exclude best_v
            new_candidates = candidates_mask & ~v_bit
            if find_indep(new_candidates, current_size):
                return True
                
            return False
            
        full_mask = (1 << n) - 1
        return not find_indep(full_mask, 0)

    def get_edges_from_mask(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Strategy 1: Construct known triangle-free graphs with small alpha.
    # The And(9) graph has 25 vertices.
    # We can try to construct a graph on 40 vertices using a "Paley-like" or "Strongly Regular" approach.
    # However, generating a valid S(3, 10) graph from scratch is hard.
    # 
    # Alternative: Use a known construction for R(3, 10) > 39.
    # The Schrijver graph (And(9) is a subgraph?) No, And(9) is the Andrasfai graph.
    # Andrasfai graph A_n is triangle-free and has alpha = 2 for n>=5? No.
    # Andrasfai graph A_n: vertices are n-tuples of 0,1,-1 with sum 0. 
    # For n=9, number of vertices is C(9, 4) + C(9, 3) * 2? 
    # Actually, the standard Andrasfai graph A_9 has 25 vertices.
    # It is triangle-free. Its independence number is 5? 
    # If alpha(A_9) = 5, we can combine two copies? No, that creates triangles or large independent sets.
    
    # Let's use a randomized local search starting from a dense random graph, 
    # but with a strong triangle-removal step and alpha-checking.
    
    # Better: Start with a known valid graph on n=30-35 and try to extend to 40.
    # Or just run a heavy local search on n=40.
    
    # Given the time limit, a robust local search on n=40 is best.
    # We need to ensure we don't get stuck in local minima with triangles.
    
    def local_search(n, max_time=100):
        """
        Local search to find triangle-free graph on n vertices with alpha < 10.
        """
        start_ls = time.time()
        
        # Initialize with a random graph
        # Probability of edge p.
        # For triangle-free, we want sparse? No, dense random has many triangles.
        # Start with a graph that is likely triangle-free.
        # e.g., Random bipartite graph? Bipartite graphs are triangle-free.
        # Max independent set in bipartite graph?
        # If we take a random bipartite graph with parts of size n/2, 
        # alpha is roughly n/2 + something. 
        # For n=40, parts of 20. Alpha could be 20+. Too big.
        # We need to add edges within parts to reduce alpha, but that creates triangles?
        # No, adding edges within parts of a bipartite graph creates triangles if we add edges carefully?
        # If we add an edge (u, v) in part 1, and u, v both connect to w in part 2, we get a triangle.
        # So we must be careful.
        
        # Strategy: Start with a random graph, remove all triangles.
        # This ensures triangle-free.
        # Then check alpha. If alpha >= 10, try to add edges to reduce alpha 
        # while maintaining triangle-free property.
        
        best_alpha = float('inf')
        best_adj = [set() for _ in range(n)]
        
        for restart in range(50):
            if time.time() > max_time:
                break
            
            # 1. Generate random graph
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.4:
                        adj[i].add(j)
                        adj[j].add(i)
                        
            # 2. Remove triangles
            # Greedy: While there is a triangle, remove a random edge from it.
            # To speed up, we can find all triangles and remove edges that break the most.
            # Simple approach:
            changed = True
            while changed:
                changed = False
                # Find a triangle
                found = False
                for i in range(n):
                    if found: break
                    for j in adj[i]:
                        if j <= i: continue
                        for k in adj[j]:
                            if k > j and k in adj[i]:
                                # Triangle i-j-k
                                # Remove a random edge
                                edge_to_remove = random.choice([(i, j), (j, k), (i, k)])
                                u, v = edge_to_remove
                                adj[u].discard(v)
                                adj[v].discard(u)
                                changed = True
                                found = True
                                break
                        if found: break
                    if found: break
                
            # 3. Check Alpha
            # Convert to bitmask for fast check
            adj_masks = [0] * n
            for i in range(n):
                mask = 0
                for j in adj[i]:
                    mask |= (1 << j)
                adj_masks[i] = mask
            
            # Check if alpha < 10
            if check_valid_bitmask(adj_masks, n, 10):
                # Valid!
                return get_edges_from_mask(adj_masks, n)
            else:
                # Invalid (alpha >= 10). 
                # Try to reduce alpha by adding edges.
                # Adding an edge (u, v) reduces alpha if u and v are in the same independent set.
                # But adding an edge must not create a triangle.
                # So u and v must not share any common neighbor.
                
                # Find all "safe" edges that can be added.
                # Safe edge (u, v) if:
                # 1. (u, v) not in adj
                # 2. N(u) and N(v) are disjoint.
                
                # We will try to add random safe edges to reduce alpha.
                for step in range(200):
                    if time.time() > max_time:
                        return best_edges if best_alpha < 10 else None # Return best found or None
                    if time.time() > deadline - 5:
                        return best_edges if best_alpha < 10 else None
                    
                    # Find a safe edge
                    safe_edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if v in adj[u]:
                                continue
                            # Check if N(u) and N(v) are disjoint
                            # (adj[u] & adj[v]) is empty
                            if not (adj[u] & adj[v]):
                                safe_edges.append((u, v))
                    
                    if not safe_edges:
                        break # No more edges can be added without creating triangles
                    
                    # Pick a random safe edge
                    u, v = random.choice(safe_edges)
                    adj[u].add(v)
                    adj[v].add(u)
                    
                    # Update masks
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
                    # Check validity
                    if check_valid_bitmask(adj_masks, n, 10):
                        return get_edges_from_mask(adj_masks, n)

        return None

    # Try n=40 first
    edges_40 = local_search(40, max_time=deadline - start_time - 5)
    if edges_40 is not None:
        return {'n': 40, 'edges': edges_40}
        
    # If 40 fails, try 39
    if time.time() < deadline - 10:
        edges_39 = local_search(39, max_time=deadline - start_time - 5)
        if edges_39 is not None:
            return {'n': 39, 'edges': edges_39}

    # Fallback: Return a known valid graph if we can construct one quickly.
    # Andrasfai graph A_9 has 25 vertices.
    # We can just return the best we found, or a trivial one.
    # But we need to maximize n.
    
    # Let's try a simpler local search for n=30 if time permits, just to have something.
    # Actually, if 40 and 39 fail, we should return the best valid graph we can find.
    # The local_search returns None if it doesn't find a valid one.
    # We should modify it to keep the best graph found (lowest alpha) even if invalid, 
    # but the task requires a VALID graph.
    
    # If no valid graph found, we must return something valid.
    # The smallest valid graph is n=1? No, alpha < 10. 
    # A complete graph K_9 has alpha=1, triangle-free? No, K_9 has triangles.
    # We need triangle-free AND alpha < 10.
    # C_5 is triangle-free, alpha=2. n=5.
    # We can always return a small valid graph.
    
    # Let's construct a valid graph on n=10 (e.g. C_5 disjoint union C_5? No, alpha=4. Valid).
    # Or just a single C_5. n=5.
    # To maximize score, we want n as large as possible.
    # If local search fails, we return a hardcoded valid graph on n=25 (And(9))?
    # Constructing And(9) explicitly is complex.
    
    # Let's just return a simple valid graph on n=10.
    # Cycle C_10 is triangle-free. Alpha(C_10) = 5. Valid.
    edges_fallback = []
    n_fallback = 10
    for i in range(n_fallback):
        edges_fallback.append([i, (i+1) % n_fallback])
        
    return {'n': n_fallback, 'edges': edges_fallback}

# EVOLVE-BLOCK-END
