# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
    forbidden_sym.discard(0)
    
    # Precompute conflict mask
    # conflict[i] is a set of j such that (i-j) % m in forbidden_sym
    conflict = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden_sym:
                conflict[i].add(j)
    
    # Precompute adjacency lists for faster iteration
    # adj[i] = list of neighbors of i in the conflict graph
    adj = [list(conflict[i]) for i in range(m)]
    
    # Precompute non-neighbors for potential use
    # Actually, let's use a bitset approach for speed if possible, or just sets.
    # With m=205, sets are fine.
    
    # Bitset representation for speed in branch and bound
    # Use Python integers as bitsets.
    # bit i is set if vertex i is in the set.
    
    # Precompute neighbor bitmasks
    neigh_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        neigh_mask[i] = mask
    
    # Precompute candidate masks for subsets? No, dynamic.
    
    # Global variables for branch and bound
    best_set = []
    best_size = 0
    deadline = time.time() + 610.0
    stop_flag = [False]
    
    # Greedy coloring heuristic for upper bound
    # We need a fast upper bound on the size of the independent set in a subgraph.
    # We can use a greedy coloring of the complement graph?
    # Or a greedy packing of cliques in the original graph (which are independent sets in the complement)?
    # Actually, for Maximum Independent Set (MIS), a common upper bound is the greedy coloring of the graph itself
    # where we color the graph such that no two adjacent vertices have the same color.
    # Wait, MIS in G is Clique in G_complement.
    # Upper bound on Clique size is the number of colors in a greedy coloring of G_complement?
    # No, the number of colors in a greedy coloring of G is an upper bound on the size of a Clique in G.
    # So for MIS in G (Clique in G_comp), we need an upper bound on Clique in G_comp.
    # This is the number of colors in a greedy coloring of G_comp.
    # Let's implement a greedy coloring of G_comp restricted to the candidate set C.
    
    # G_comp edges: i,j in C are adjacent in G_comp if NOT adjacent in G.
    # i.e., j not in adj[i].
    
    def greedy_color_bound(candidates_mask):
        """
        Compute a greedy coloring upper bound for the clique size in G_comp restricted to candidates_mask.
        This is equivalent to the independent set size in G restricted to candidates_mask.
        Returns the number of colors used.
        """
        # Extract vertices
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            verts.append(i)
            temp ^= lsb
        
        if not verts:
            return 0
            
        # Greedy coloring of G_comp induced by verts
        # Colors are assigned such that no two adjacent in G_comp have same color.
        # Adjacent in G_comp means NOT adjacent in G.
        # So we want to assign colors to verts such that if u,v in verts and v in adj[u] (adjacent in G),
        # then color[u] != color[v]?
        # Wait.
        # Clique in G_comp corresponds to Independent Set in G.
        # Upper bound on Clique size is obtained by coloring the graph where we are finding the clique.
        # So we color G_comp.
        # In G_comp, u and v are adjacent if NOT (v in adj[u]).
        # So in G_comp, u and v are NOT adjacent if v in adj[u].
        # Greedy coloring: assign smallest color not used by neighbors in G_comp.
        # Neighbors in G_comp are those v in verts such that NOT (v in adj[u]).
        
        color_of = {}
        next_color = 0
        # Order matters. Randomize or use fixed order?
        # For an upper bound, any valid coloring works. The number of colors is the bound.
        # A better (tighter) bound is obtained with a better coloring order.
        # Let's try ordering by degree in G_comp (which is (len(verts)-1 - deg_G(u))).
        # High degree in G_comp means many non-neighbors in G.
        
        # Let's just use the natural order or random order for simplicity first.
        # Actually, to make the bound tighter, we should try to minimize colors.
        
        # We will iterate through vertices and assign the smallest possible color.
        # To check if color c is available for u, we check if any neighbor of u in G_comp has color c.
        
        # This is O(V^2) per call. With V up to 205, and many calls, this might be slow.
        # Let's optimize or use a simpler bound if too slow.
        # Simpler bound: Max degree + 1? No.
        # Let's stick with greedy coloring but optimize the check.
        
        # Actually, there is a simpler upper bound for MIS:
        # Sum of 1/(deg(v)+1) for v in C? No.
        
        # Let's use the greedy coloring but cache or optimize.
        # Given the small size (205), let's try it.
        
        # To speed up, we can precompute the non-neighbor masks.
        # non_neigh_mask[i] = mask of vertices NOT in adj[i] (and not i itself? usually simple graph).
        # In G_comp, neighbors of i are {v in C | v != i and v not in adj[i]}.
        
        # Let's precompute non_adj_mask[i] for all i in 0..204.
        # non_adj_mask[i] = ((1 << 205) - 1) ^ (1 << i) ^ neigh_mask[i]
        
        # Inside greedy_color_bound:
        # For each u in verts:
        #   forbidden_colors = set()
        #   for v in verts:
        #      if v != u and (non_adj_mask[u] >> v) & 1:
        #          if v in color_of:
        #              forbidden_colors.add(color_of[v])
        #   color_of[u] = smallest non-negative int not in forbidden_colors
        
        # This is O(V^2).
        
        # Let's implement it.
        
        # Optimization: Sort vertices by degree in G_comp descending?
        # deg_comp(u) = |C| - 1 - deg_G(u, C)
        # deg_G(u, C) = popcount(neigh_mask[u] & candidates_mask)
        
        # Let's just do it.
        
        # We need non_adj_mask precomputed globally.
        
        # Let's move non_adj_mask computation outside.
        
        # For now, return a loose bound if this is too complex to inline efficiently.
        # A very fast bound: len(verts)
        # A better bound: max clique size of G_comp?
        
        # Let's try a simple greedy coloring.
        
        # To make it fast, we'll use the fact that m is small.
        
        # We'll use a list for colors.
        
        # Actually, let's use a different bound:
        # The maximum independent set size is at most the size of a maximum matching? No.
        
        # Let's use the bound from the greedy algorithm itself:
        # If we greedily pick vertices, we get a lower bound.
        # For upper bound, we can use the fact that each vertex in the independent set can "cover" itself and its neighbors.
        # This is a packing argument.
        
        # Let's stick to the greedy coloring of G_comp.
        
        # Precompute non_adj_mask globally to speed this up.
        
        pass # Placeholder, will implement in main body

    # Precompute non_adj_mask
    all_mask = (1 << m) - 1
    non_adj_mask = [0] * m
    for i in range(m):
        # Non-neighbors in G are those not in neigh_mask[i] and not i.
        non_adj_mask[i] = all_mask ^ (1 << i) ^ neigh_mask[i]
    
    def greedy_color_bound(candidates_mask):
        """
        Greedy coloring of G_comp restricted to candidates_mask.
        Returns number of colors.
        """
        # Extract vertices
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            verts.append(i)
            temp ^= lsb
        
        n = len(verts)
        if n == 0:
            return 0
            
        # Order vertices by degree in G_comp descending (to potentially get tighter bound? or ascending?)
        # Usually, coloring with high degree first is better for minimizing colors.
        # deg_comp(u) = n - 1 - popcount(neigh_mask[u] & candidates_mask)
        
        # Calculate degrees
        degrees = []
        for u in verts:
            deg_g = bin(neigh_mask[u] & candidates_mask).count('1')
            deg_comp = n - 1 - deg_g
            degrees.append((deg_comp, u))
        
        # Sort by degree descending
        degrees.sort(reverse=True)
        ordered_verts = [u for _, u in degrees]
        
        color_of = {}
        max_color = 0
        
        for u in ordered_verts:
            # Find colors used by neighbors in G_comp
            # Neighbors in G_comp are v in verts such that (non_adj_mask[u] >> v) & 1
            # Which means v is NOT in adj[u].
            
            used_colors = set()
            # Iterate through other vertices
            # To speed up, we can check the mask of neighbors in G_comp
            # neigh_comp_mask_u_in_verts = non_adj_mask[u] & candidates_mask & ~(1 << u)
            # Then iterate through bits in that mask.
            
            neigh_comp_mask = non_adj_mask[u] & candidates_mask & ~(1 << u)
            
            temp_mask = neigh_comp_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                if v in color_of:
                    used_colors.add(color_of[v])
                temp_mask ^= lsb
                
            # Find smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            color_of[u] = c
            if c > max_color:
                max_color = c
            c += 1 # Next color to check? No, just max_color+1 is upper bound for next.
            
        return max_color + 1

    def branch_and_bound(candidates_mask, current_size, current_set):
        """
        Recursive branch and bound for Maximum Independent Set.
        """
        nonlocal best_set, best_size, stop_flag
        
        if stop_flag[0]:
            return
            
        if time.time() > deadline:
            stop_flag[0] = True
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        # Upper bound
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= best_size:
            return
            
        # Choose a vertex to branch on
        # Heuristic: Choose vertex with highest degree in G restricted to candidates?
        # Or lowest?
        # Let's choose the vertex with the highest degree in the induced subgraph.
        # High degree means it conflicts with many, so branching on it might prune more?
        # Actually, standard MIS heuristics often pick the vertex with the highest degree.
        
        # Calculate degrees
        temp = candidates_mask
        best_v = -1
        best_deg = -1
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = bin(neigh_mask[v] & candidates_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New candidates = candidates_mask & ~neigh_mask[best_v] & ~(1 << best_v)
        new_mask = candidates_mask & ~neigh_mask[best_v] & ~(1 << best_v)
        current_set.append(best_v)
        branch_and_bound(new_mask, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_mask = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_mask, current_size, current_set)

    # Initial upper bound using greedy coloring on all vertices
    initial_bound = greedy_color_bound((1 << m) - 1)
    
    # We can also get a good initial lower bound using greedy construction
    # Let's run a few random greedy constructions to get a good start
    for _ in range(100):
        order = list(range(m))
        random.shuffle(order)
        S = []
        S_set = set()
        for x in order:
            # Check if x conflicts with any in S_set
            # Using bitsets
            # mask of S_set
            # Let's just use sets for this quick greedy
            ok = True
            for s in S_set:
                if x in conflict[s]:
                    ok = False
                    break
            if ok:
                S.append(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Now run branch and bound
    # Start with all vertices
    # We need to be careful with recursion depth. m=205, depth 205 is fine for Python?
    # Default recursion limit is 1000.
    sys.setrecursionlimit(5000)
    
    # Run B&B
    # To avoid getting stuck in a bad branch order, we can randomize the vertex selection slightly?
    # Or just use the degree heuristic.
    
    # Let's run it.
    # We might need to restart B&B with different vertex orderings if it's too slow or suboptimal.
    # But B&B is exact, so it will find the optimum if it finishes.
    # The bound is the key.
    
    # Let's try running it once.
    # If it takes too long, we return the best found so far.
    
    # To make it more robust, we can try multiple times with different random perturbations in vertex ordering?
    # Or just rely on the degree heuristic.
    
    # Let's run.
    # Note: The greedy_color_bound is called many times. It might be a bottleneck.
    # If it's too slow, we can simplify the bound to just len(candidates) or a simpler heuristic.
    
    # Let's try it.
    
    # We'll run B&B.
    # If stop_flag is set, we return best_set.
    
    # To improve the chance of finding a good solution quickly, we can also do a local search on the best_set found.
    
    # Run B&B
    branch_and_bound((1 << m) - 1, 0, [])
    
    # If B&B didn't finish or found a small set, we can try local search on the best set found.
    # But B&B is exact, so if it finishes, best_set is optimal.
    # If it times out, best_set is the best found in the partial search.
    
    # Let's also run a local search on the best_set to see if we can improve it further
    # (in case B&B pruned incorrectly or we want to polish).
    # Actually, if B&B is correct, it finds the max.
    # But if it times out, it might not have explored all branches.
    # So local search on the current best might help.
    
    def local_search(S, S_set):
        improved = True
        while improved and not stop_flag[0]:
            improved = False
            # Try to add any possible element
            for x in range(m):
                if x not in S_set:
                    ok = True
                    for s in S_set:
                        if x in conflict[s]:
                            ok = False
                            break
                    if ok:
                        S.append(x)
                        S_set.add(x)
                        improved = True
            # Try remove+add
            if not improved:
                for u in list(S):
                    S_copy = list(S)
                    S_set_copy = set(S_set)
                    S_copy.remove(u)
                    S_set_copy.remove(u)
                    for x in range(m):
                        if x not in S_set_copy:
                            ok = True
                            for s in S_set_copy:
                                if x in conflict[s]:
                                    ok = False
                                    break
                            if ok:
                                S_copy.append(x)
                                S_set_copy.add(x)
                    if len(S_copy) > len(S):
                        S = S_copy
                        S_set = S_set_copy
                        improved = True
        return S, S_set
    
    # Run local search on best_set
    if best_set:
        best_set, best_set_set = local_search(best_set, set(best_set))
        if len(best_set) > best_size:
            best_size = len(best_set)
            
    # If best_set is empty (should not happen), return [0]
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
