import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    n = 256
    # 1. Identify valid vertices (no self-loop)
    squares = set()
    k = 1
    while k * k <= 2 * n:
        squares.add(k * k)
        k += 1
        
    valid_vertices = []
    for i in range(1, n + 1):
        if 2 * i not in squares:
            valid_vertices.append(i)
            
    m = len(valid_vertices)
    # Map vertex value to index in valid_vertices
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build adjacency bitmasks
    adj = [0] * m
    for i in range(m):
        vi = valid_vertices[i]
        for j in range(i + 1, m):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Global best
    best_size = 0
    best_set = []
    
    # Helper for greedy coloring upper bound
    # This is a simplified version of Tomita's bound.
    # It estimates the size of the maximum independent set in the subgraph induced by candidates.
    # We use a greedy coloring of the complement graph (which is the conflict graph here) 
    # Wait, MIS in G is equivalent to Clique in G_complement.
    # But we are working with Independent Set directly.
    # A common bound for MIS is the greedy coloring of the conflict graph? No.
    # The size of a maximum independent set is at most the number of colors in a greedy coloring of the graph?
    # Actually, for MIS, we can use the bound: sum over connected components? No.
    # A simple effective bound: Greedy Independent Set size.
    # Let's use a fast greedy heuristic to estimate the upper bound.
    
    def greedy_is_size(candidates_mask):
        size = 0
        rem = candidates_mask
        while rem:
            # Pick a vertex with max degree in rem? Or just any?
            # For speed, just pick the first one? No, that's a lower bound for the set, not an upper bound for the remaining.
            # We need an UPPER bound on the size of MIS in the subgraph defined by candidates_mask.
            # One such bound is the clique cover number, which is hard.
            # Another is: sort vertices by degree, pick highest, remove it and its neighbors.
            # This gives a valid independent set, so it's a LOWER bound on the optimal MIS size?
            # No, we want to prune if current_size + upper_bound <= best.
            # So we need an upper bound.
            # The trivial upper bound is popcount.
            # A better upper bound: The size of a maximum matching? No.
            # Let's use the property that in a graph with max degree D, the MIS size is at least n/(D+1).
            # That's a lower bound.
            
            # Let's stick to a strong pruning strategy:
            # 1. If current_size + popcount(candidates) <= best_size: prune.
            # 2. Use a greedy coloring bound on the COMPLEMENT graph?
            # Actually, the standard "Tomita" algorithm for Max Clique uses coloring.
            # Since MIS(G) = MaxClique(G_complement), we can apply Tomita's coloring bound on the complement graph.
            # However, building the complement graph is O(N^2).
            
            # Alternative: Use the "Max Degree" bound.
            # Let d_max be the max degree in the subgraph.
            # Upper bound on MIS is not directly d_max.
            
            # Let's try a simple recursive bound with memoization? No, state space too large.
            
            # Let's use a greedy heuristic to find a large independent set quickly to update best_size,
            # and rely on the popcount bound with good vertex ordering.
            # The previous failure was likely due to bad vertex ordering or lack of a tighter bound.
            
            # Let's implement a simple greedy coloring bound for the MIS problem directly?
            # Actually, there is a bound: The size of the MIS is at most the sum of the sizes of the connected components' MIS?
            
            # Let's try this:
            # Upper bound = popcount(candidates) is too loose.
            # Let's use the "independent set" greedy as a LOWER bound to update best, 
            # and keep the popcount upper bound but with a much better vertex selection strategy.
            pass
            return 0 # Placeholder, we will implement a real bound below

    # Real Upper Bound Strategy:
    # We will use a greedy coloring on the graph of candidates to estimate the clique number of the complement?
    # Too complex.
    
    # Let's use a different approach: Branch and Bound with "Max Degree" selection and a tight time check.
    # The key improvement is the vertex ordering and the bound.
    # Bound: current_size + popcount(candidates_mask) is weak.
    # Better Bound: 
    # Let's calculate the greedy independent set size of the candidates. 
    # This is a LOWER bound on the optimal solution size for that subgraph.
    # It doesn't help prune directly unless we use it to update best.
    
    # Let's use the "Tomita" style bound for MIS.
    # We can transform the problem to Max Clique in the complement graph.
    # But we don't have the complement graph built.
    
    # Let's try a simpler, robust heuristic:
    # 1. Greedy Local Search to get a good initial best_size.
    # 2. Branch and Bound with popcount bound, but selecting vertices with highest degree first.
    
    # Initial Solution: Greedy
    def get_greedy_solution():
        sol = set()
        remaining = set(range(m))
        while remaining:
            # Pick vertex with min degree in remaining?
            # Or max degree?
            # For MIS, picking a vertex with high degree removes many neighbors, 
            # so it's a "risky" pick but reduces the problem size fast.
            # Picking a vertex with low degree is "safe" but leaves more vertices.
            # Standard greedy for MIS: pick vertex with max degree, add to set, remove it and neighbors.
            # This yields a maximal independent set.
            best_v = -1
            max_deg = -1
            for v in remaining:
                # Degree in remaining
                deg = 0
                # Count neighbors in remaining
                # adj[v] & mask_of_remaining
                # We need a mask for remaining.
                pass
            # This is slow.
            
            # Let's just pick a random or first valid vertex for speed in initialization?
            # No, let's use the degrees precomputed.
            # We need dynamic degrees.
            
            # Let's just use a simple heuristic:
            # Iterate through valid vertices, if it doesn't conflict with current set, add it.
            # This is O(N^2). N=150. 150^2 = 22500. Fast.
            
            current_set = []
            used = [False] * m
            for i in range(m):
                conflict = False
                for j in current_set:
                    if (adj[i] >> j) & 1:
                        conflict = True
                        break
                if not conflict:
                    current_set.append(i)
                    used[i] = True
            return current_set

    initial_sol = get_greedy_solution()
    best_size = len(initial_sol)
    best_set = initial_sol[:]
    
    # DFS
    def dfs(candidates_mask, current_size, current_set_indices):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        # Bound
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set_indices[:]
            return
            
        # Select vertex
        # Heuristic: Pick vertex with max degree in the subgraph defined by candidates_mask
        v = -1
        max_deg = -1
        # Iterate through bits in candidates_mask
        temp_mask = candidates_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            # Calculate degree of idx in candidates_mask
            # Neighbors of idx are in adj[idx]
            # Intersection with candidates_mask
            deg = bin(adj[idx] & candidates_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_mask ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # Remove v and its neighbors from candidates
        new_candidates_inc = candidates_mask & ~(adj[v] | (1 << v))
        current_set_indices.append(v)
        dfs(new_candidates_inc, current_size + 1, current_set_indices)
        current_set_indices.pop()
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_candidates_exc = candidates_mask & ~(1 << v)
        dfs(new_candidates_exc, current_size, current_set_indices)
        
    initial_mask = (1 << m) - 1
    dfs(initial_mask, 0, [])
    
    result = [valid_vertices[i] for i in best_set]
    
    if not result:
        if valid_vertices:
            result = [valid_vertices[0]]
        else:
            result = [1]
            
    return result
