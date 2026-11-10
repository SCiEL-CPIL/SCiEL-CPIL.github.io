"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring (no Property B).
    Strategy:
    1. Use a fast backtracking 2-colorability checker with unit propagation.
    2. Generate candidate hypergraphs by adding edges to an empty set until non-2-colorable.
    3. Minimize the candidate by greedily removing non-critical edges.
    4. Repeat for multiple random orders and n values to find the smallest set.
    """
    import random
    import time
    from itertools import combinations
    from collections import defaultdict

    def is_2_colorable(n, edges):
        """
        Check if a 5-uniform hypergraph on n vertices is 2-colorable using backtracking with unit propagation.
        Returns True if a valid 2-coloring exists, False otherwise.
        """
        if not edges:
            return True
            
        # Precompute incidence: vertex -> list of edge indices
        vertex_edges = defaultdict(list)
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
                
        colors = [-1] * n
        
        def propagate(queue):
            """
            Apply unit propagation.
            Returns False if a contradiction is found.
            Modifies 'colors' and 'queue' in place.
            """
            while queue:
                v = queue.pop()
                c = colors[v]
                if c == -1:
                    continue # Should not happen if we only push assigned vertices
                
                for e_idx in vertex_edges[v]:
                    edge = edges[e_idx]
                    # Count colors in this edge
                    count_0 = 0
                    count_1 = 0
                    unassigned = []
                    
                    for u in edge:
                        if colors[u] == 0:
                            count_0 += 1
                        elif colors[u] == 1:
                            count_1 += 1
                        else:
                            unassigned.append(u)
                            
                    # Check for immediate contradiction
                    if count_0 == 5 or count_1 == 5:
                        return False
                    
                    # Unit propagation: if 4 are 0, the 5th must be 1 (and vice versa)
                    if len(unassigned) == 1:
                        u = unassigned[0]
                        if count_0 == 4:
                            new_c = 1
                        elif count_1 == 4:
                            new_c = 0
                        else:
                            # 3 or fewer assigned, no unit clause
                            continue
                        
                        if colors[u] != -1:
                            if colors[u] != new_c:
                                return False
                        else:
                            colors[u] = new_c
                            queue.append(u)
            return True

        def backtrack(idx):
            """
            Recursive backtracking.
            idx is a heuristic index for vertex selection, or we just pick first unassigned.
            For simplicity and speed with n<=12, picking first unassigned or most constrained is good.
            """
            # Find an unassigned vertex
            v = -1
            # Heuristic: pick vertex with most assigned neighbors in its edges?
            # For now, simple: first unassigned.
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                return True # All assigned, no conflict found
            
            # Try color 0
            colors[v] = 0
            # Copy queue? No, we can use a stack for propagation that we manage carefully.
            # Since n is small, we can just re-propagate from scratch or maintain a queue.
            # Let's use a simple recursive step where we propagate immediately.
            
            # To make propagation safe with backtracking, we need to revert changes.
            # Easier: copy the colors array. n <= 12, so 12 ints copy is fast.
            colors_backup = colors[:]
            q = [v]
            if propagate(q):
                if backtrack(idx + 1):
                    return True
            colors[:] = colors_backup
            
            # Try color 1
            colors[v] = 1
            colors_backup = colors[:]
            q = [v]
            if propagate(q):
                if backtrack(idx + 1):
                    return True
            colors[:] = colors_backup
            
            return False

        # Initialize
        # We can add some initial propagation if there are any constraints? No, all unassigned.
        # Just start backtracking.
        return backtrack(0)

    def minimize_hypergraph(n, initial_edges):
        """
        Greedily remove edges that do not restore 2-colorability.
        """
        current_edges = initial_edges[:]
        # Shuffle to randomize removal order
        random.shuffle(current_edges)
        
        changed = True
        while changed:
            changed = False
            # We iterate through a copy of the indices to allow removal
            # But we need to be careful. Let's just iterate through the list and try to remove.
            # To be efficient, we check from the end or shuffle.
            
            # We'll try to remove each edge. If we remove one, we restart the loop.
            # This is O(E^2 * CheckTime). With E < 100 and CheckTime < 1ms, this is fine.
            
            for i in range(len(current_edges) - 1, -1, -1):
                edge_to_remove = current_edges[i]
                # Create a list without this edge
                # Optimization: use a set for O(1) check? No, list removal is O(E).
                # Since E is small, simple list comprehension is fine.
                temp_edges = [e for j, e in enumerate(current_edges) if j != i]
                
                if not is_2_colorable(n, temp_edges):
                    # Still non-2-colorable, so we can remove it
                    current_edges.pop(i)
                    changed = True
                    break # Restart loop with new list
        return current_edges

    best_edges = None
    best_n = 0
    start_time = time.time()
    time_limit = 100 # seconds
    
    # Try n from 6 to 11
    # n=6,7,8 might be too small to have non-2-colorable graphs with few edges?
    # Actually, m(5) is the minimum number of edges.
    # Known: m(5) <= 51.
    # Let's try n=9, 10, 11 as they are likely to yield good results.
    # We will do multiple random trials for each n.
    
    for n in [9, 10, 11]:
        if time.time() - start_time > time_limit:
            break
            
        all_possible_edges = [list(e) for e in combinations(range(n), 5)]
        num_edges_total = len(all_possible_edges)
        
        # Number of trials depends on n and time
        # For n=9, 126 edges.
        # For n=10, 252 edges.
        
        trials = 20 if n == 9 else 10
        
        for _ in range(trials):
            if time.time() - start_time > time_limit:
                break
                
            # Shuffle edges to get a random order
            shuffled_edges = all_possible_edges[:]
            random.shuffle(shuffled_edges)
            
            # Build up until non-2-colorable
            current_edges = []
            # We can add a few edges at a time or one by one.
            # One by one is safer for finding minimal, but slower.
            # Let's add in batches of 1 to ensure minimality potential.
            # Actually, adding one by one and checking is expensive.
            # Alternative: Start with a random subset of size K, check.
            # If 2-colorable, add more. If not, minimize.
            
            # Let's try starting with a random set of size ~ 30-50.
            # If it's 2-colorable, add random edges until it's not.
            # If it's not, minimize.
            
            # Start with a random permutation of all edges
            # Add edges until non-2-colorable
            
            # To speed up, we can skip the "add one by one" for the first few.
            # Just pick a random subset of size M.
            # If 2-colorable, it's useless.
            # If not, minimize.
            
            # What M?
            # For n=9, K5_9 has 126 edges.
            # Random graphs become non-2-colorable around density 0.5?
            # Let's try M = 40, 50, 60.
            
            found_candidate = False
            # Try different starting sizes
            for start_size in [40, 50, 60, 70]:
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random subset of size start_size
                # To ensure variety, we use the shuffled list
                candidate = shuffled_edges[:start_size]
                
                if not is_2_colorable(n, candidate):
                    # It is non-2-colorable
                    # Minimize it
                    minimized = minimize_hypergraph(n, candidate)
                    
                    if best_edges is None or len(minimized) < len(best_edges):
                        best_edges = minimized
                        best_n = n
                        
                    found_candidate = True
                    break # If we found one, try next trial (different shuffle)
            
            # If we didn't find a non-2-colorable graph with those sizes, 
            # we could try adding edges one by one, but that's slow.
            # Given the density, start_size=70 should almost always be non-2-colorable for n=9,10.
            # If not, we might miss it, but multiple trials help.

    # Fallback if nothing found (should not happen)
    if best_edges is None:
        # Use the old greedy method on K5_9 as fallback?
        # Or just return a known valid one?
        # Let's just do a quick greedy on K5_9
        n = 9
        all_edges = [list(e) for e in combinations(range(n), 5)]
        random.shuffle(all_edges)
        current = all_edges[:]
        # Quick minimize
        changed = True
        while changed:
            changed = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    break
                temp = [e for j, e in enumerate(current) if j != i]
                if not is_2_colorable(n, temp):
                    current.pop(i)
                    changed = True
                    break
        best_edges = current
        best_n = n

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
