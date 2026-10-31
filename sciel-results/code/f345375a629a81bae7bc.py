import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # 1. Generate Pool of Weight-4 Vectors
    # Norm squared = 4. Norm = 2.
    # Compatibility condition: dist^2 >= 4 => 4+4 - 2*u.v >= 4 => u.v <= 2.
    # So, vectors are compatible if dot product <= 2.
    
    candidates = []
    indices = range(D)
    # Weight 4: 4 non-zero entries, each +/-1
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    if time.time() > deadline - 10:
        return candidates[:10] # Fallback

    # 2. Compute Dot Products and Build Compatibility Matrix
    # We want to find the Maximum Clique in the graph where edges represent compatibility.
    dots = candidates @ candidates.T
    
    # Compatibility: i != j and dots[i, j] <= 2
    # Note: diagonal is 4, which is > 2, so diagonal is naturally 0 in compatibility if we just check <= 2?
    # Wait, if we check dots <= 2, diagonal (4) is False. So A[i,i] will be 0.
    # This is exactly what we want for a simple adjacency matrix for Max Clique (no self-loops).
    
    comp_matrix = (dots <= 2).astype(np.int8)
    
    # Ensure symmetry (it should be symmetric by construction, but good to be sure)
    # comp_matrix = np.maximum(comp_matrix, comp_matrix.T) # Not needed, dots is symmetric
    
    # 3. Tomita's Algorithm for Maximum Clique
    # We represent the graph using bitsets for speed.
    # n_cands is ~5280. This fits in an array of uint64s (5280 / 64 = 83 words).
    # However, Python's int is arbitrary precision, so we can use Python ints as bitsets directly.
    # This is often faster than numpy for bit operations on large integers in pure Python loops if we are careful,
    # but numpy is better for vectorized ops. Tomita is recursive and branchy, so Python ints might be okay if we minimize overhead.
    # Let's use Python ints for the bitsets of the candidate set.
    
    # Convert comp_matrix rows to Python ints (bitsets)
    # Bit i is 1 if vertex i is in the set.
    # We need to map row index to bit position.
    
    # Precompute bitsets for each vertex's neighbors in the compatibility graph
    # neighbor_bits[i] is a Python int where bit j is 1 if comp_matrix[i, j] == 1
    
    neighbor_bits = [0] * n_cands
    for i in range(n_cands):
        # Extract row i
        # We can use numpy to get the indices where comp_matrix[i] == 1
        # But converting to a Python int from a boolean array is O(N).
        # Doing this N times is O(N^2). 5280^2 ~ 27M ops. This is fast in Python if vectorized?
        # Let's try to do it efficiently.
        
        # Get indices of compatible vertices
        compat_indices = np.nonzero(comp_matrix[i])[0]
        
        # Build the integer bitset
        # This loop might be slow if done naively for 5280 vertices.
        # Alternative: Use numpy to pack bits?
        # Or just accept the O(N^2) setup cost if it's < 1s.
        
        b = 0
        for j in compat_indices:
            b |= (1 << j)
        neighbor_bits[i] = b

    # Check time
    if time.time() > deadline - 5:
        # Fallback to a simple greedy on the compatibility graph
        # Greedy Clique: Pick vertex with highest degree, add to clique, remove neighbors.
        # Actually, let's just return a small set if we are out of time.
        return candidates[:10]

    # Tomita's Max Clique Algorithm
    # Parameters:
    # U: set of candidates (bitmask)
    # P: set of vertices processed (bitmask) - used for coloring heuristic
    # C: current clique (list of vertices)
    # best_clique: global best clique found
    
    best_clique = []
    
    # Helper: Get degree (popcount of neighbor_bits[i] & U)
    def popcount(x):
        return x.bit_count()
        
    # Helper: Coloring heuristic
    # We need to color the subgraph induced by U to get an upper bound.
    # Standard Tomita uses a greedy coloring of the complement? No, it colors the graph itself to bound the clique size.
    # Wait, for Max Clique, the upper bound is the clique number.
    # The greedy coloring of the *graph* gives a lower bound on the chromatic number, which is an upper bound on the clique number?
    # No. Chromatic number >= Clique number.
    # So if we color the graph, the number of colors used is an upper bound on the clique size?
    # Yes, because a clique requires distinct colors.
    # So, we color the subgraph induced by U. The max color assigned is the upper bound.
    
    def color_graph(U_mask):
        # Returns a list of colors for vertices in U_mask, and the max color used.
        # Vertices in U_mask are processed in some order (e.g., degree order).
        # We need an ordering. Tomita usually orders by degree in the remaining graph.
        
        # Get list of vertices in U
        # Extracting list from bitmask is O(N).
        # To speed this up, we can maintain the list externally or use a trick.
        # For now, let's just iterate through all bits.
        
        # Optimization: We can pre-sort vertices by global degree to have a static order,
        # but Tomita's coloring is dynamic on the subgraph.
        
        # Let's implement a simple greedy coloring.
        # We need to iterate vertices in U.
        # To avoid O(N) extraction every time, we can pass a list of active vertices?
        # But the bitmask is the canonical representation.
        
        # Let's use a helper to get active vertices.
        active = []
        temp = U_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            active.append(idx)
            temp ^= lsb
            
        if not active:
            return [], 0
        
        # Sort by degree in U (descending) for better coloring?
        # Tomita's algorithm usually uses a static order or dynamic.
        # Let's sort by degree in the subgraph U.
        def deg_in_u(v):
            return popcount(neighbor_bits[v] & U_mask)
            
        active.sort(key=deg_in_u, reverse=True)
        
        colors = {}
        max_color = 0
        # Greedy coloring
        for v in active:
            # Find smallest available color
            used_colors = set()
            # Check neighbors of v in U
            neigh_mask = neighbor_bits[v] & U_mask
            # Iterate neighbors
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                n_idx = lsb.bit_length() - 1
                if n_idx in colors:
                    used_colors.add(colors[n_idx])
                temp_neigh ^= lsb
                
            # Find smallest positive integer not in used_colors
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
            
        # Return colors dict and max_color
        return colors, max_color

    def expand(U_mask, clique):
        nonlocal best_clique
        
        # If U is empty, check if current clique is better
        if U_mask == 0:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Check time
        if time.time() > deadline:
            return

        # Coloring heuristic to get upper bound
        colors, max_color = color_graph(U_mask)
        
        # If max_color + len(clique) <= len(best_clique), prune
        if len(clique) + max_color <= len(best_clique):
            return
        
        # Process vertices in U in reverse order of color (highest color first)
        # This is a key part of Tomita's algorithm for efficiency.
        # We need to group vertices by color.
        
        # Re-extract active vertices and sort by color (descending)
        active = []
        temp = U_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            active.append(idx)
            temp ^= lsb
            
        # Sort by color descending
        active.sort(key=lambda x: colors[x], reverse=True)
        
        for v in active:
            # Prune if potential clique size is not better
            if len(clique) + colors[v] <= len(best_clique):
                break # Since sorted by color desc, remaining will have lower/equal color
            
            # Add v to clique
            clique.append(v)
            
            # New U is intersection of current U and neighbors of v
            new_U = U_mask & neighbor_bits[v]
            # Remove v from new_U (though neighbor_bits[v] doesn't contain v, so it's already removed)
            
            expand(new_U, clique)
            
            # Remove v from clique
            clique.pop()
            
            # Remove v from U (for the next iteration of the loop, U should not contain v)
            U_mask &= ~(1 << v)
            
            # Update colors? 
            # Tomita's algorithm updates the coloring of the remaining vertices in U 
            # after removing v. This is complex. 
            # A simpler version just re-colors or uses the initial coloring for pruning.
            # The "reverse search" variant updates colors. 
            # For simplicity and robustness, we will rely on the initial coloring for pruning 
            # and the structural recursion. This is less optimal but much simpler to implement correctly.
            # Actually, the standard Tomita algorithm *does* update colors.
            # Let's stick to the simpler "greedy coloring at each node" approach which is still a valid B&B,
            # just potentially less tight bounds.
            
            # Important: The loop iterates over `active` which was derived from the original U.
            # We are modifying U_mask locally.
            # The condition `if len(clique) + colors[v] <= len(best_clique)` uses the *initial* color of v.
            # This is a valid upper bound because removing vertices from U cannot increase the clique number of the remaining subgraph relative to the original coloring constraints?
            # Actually, the color of v in the *original* subgraph is an upper bound on the clique size of the subgraph induced by {v} U N(v).
            # So the pruning is valid.

    # Initialize
    # Start with all vertices in U
    initial_U = (1 << n_cands) - 1
    # We might want to order the initial vertices by degree to find large cliques early
    # Tomita's algorithm usually picks the first vertex in the sorted order to branch on.
    
    # Let's sort vertices by degree descending to start with high-degree nodes
    # This helps find large cliques early, improving the lower bound for pruning.
    
    # Get degrees
    degrees = [popcount(neighbor_bits[i]) for i in range(n_cands)]
    order = np.argsort(degrees)[::-1]
    
    # We can't easily pass an ordered list to the bitmask-based recursion without changing the logic.
    # The recursion picks vertices from U. 
    # To bias towards high-degree vertices, we can simply start the recursion 
    # with a specific vertex, or modify the `active` sorting in `expand`.
    # In `expand`, we sort by color. 
    # Let's just run it. The coloring heuristic naturally prioritizes high-degree nodes 
    # (they tend to get higher colors or block more colors).
    
    # One optimization: If we find a clique of size K, we can prune branches that can't exceed K.
    # We start with an empty clique.
    
    # To speed up, let's do a greedy initial solution to set a good lower bound.
    # Greedy Max Clique on compatibility graph:
    current_clique = []
    U_greedy = initial_U
    while U_greedy:
        # Pick vertex with highest degree in U_greedy
        best_v = -1
        best_deg = -1
        temp = U_greedy
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            deg = popcount(neighbor_bits[idx] & U_greedy)
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            temp ^= lsb
            
        if best_v == -1:
            break
            
        current_clique.append(best_v)
        # Update U_greedy: keep only neighbors of best_v
        U_greedy &= neighbor_bits[best_v]
        
    best_clique = current_clique[:]
    
    # Now run the B&B
    # We need to be careful with recursion depth. n_cands is 5280.
    # Python recursion limit is 1000 by default. We need to increase it or use iterative stack.
    sys.setrecursionlimit(10000)
    
    # We will call expand with the full set.
    # However, the standard Tomita algorithm branches on one vertex at a time from the sorted list.
    # My `expand` function above iterates through ALL vertices in U and branches on each.
    # This is correct but might be slow if the sorting is expensive.
    # Let's try it.
    
    # To ensure we explore the most promising branches first, 
    # the sorting by color in `expand` is crucial.
    
    # Let's run the search
    # Note: This might take a while. We have a deadline guard inside `expand`.
    
    expand(initial_U, [])
    
    if time.time() > deadline - 2:
         # If we are close to deadline, return what we have
         pass
         
    if not best_clique:
        # Fallback
        return candidates[:1]
        
    result_vectors = candidates[best_clique]
    return result_vectors

# EVOLVE-BLOCK-END
