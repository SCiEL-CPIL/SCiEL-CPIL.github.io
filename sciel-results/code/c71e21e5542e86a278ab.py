import numpy as np
import time
import random
from functools import lru_cache

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses a branch-and-bound approach with greedy coloring upper bounds for Maximum Independent Set.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # Build conflict masks
    # conflict_mask[i] is an integer where bit j is set if i conflicts with j (i.e., j is a neighbor of i)
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    # Precompute list of vertices for quick access
    vertices = list(range(m))
    
    # Global best solution found so far
    best_size = 0
    best_solution_mask = 0
    
    # Helper to check if a mask is valid (independent set)
    def is_valid(mask):
        # This is expensive if done from scratch, but we maintain validity in B&B
        # We just need to ensure no internal conflicts.
        # Actually, B&B maintains the property that 'selected' is an independent set.
        # The candidates are those that don't conflict with selected.
        pass

    # Greedy Coloring for Upper Bound
    # We color the graph induced by 'candidates' mask.
    # The number of colors is an upper bound on the size of the MIS in that subgraph.
    def greedy_coloring(candidates_mask):
        # Extract vertices
        verts = []
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            verts.append(v)
            cm ^= lsb
        
        if not verts:
            return 0
            
        colors = {}
        # Sort by degree in the candidate subgraph? Or just process in order.
        # Sorting by degree (descending) often gives tighter bounds.
        # Degree in subgraph: count neighbors in candidates_mask
        degrees = []
        for v in verts:
            neigh = conflict_mask[v] & candidates_mask
            deg = bin(neigh).count('1')
            degrees.append((deg, v))
        
        degrees.sort(reverse=True)
        
        color_idx = 0
        color_class = 0
        uncolored = set(verts)
        
        # Simple greedy coloring
        # We can do a more sophisticated one (DSATUR) but this might be enough for 205 nodes.
        # Let's try a basic greedy: pick vertex with highest degree, assign smallest available color.
        
        # Actually, for B&B, we just need the number of colors.
        # A faster heuristic for upper bound is often just the number of vertices in the largest clique in the complement? No.
        # The number of colors in a proper coloring is an upper bound on the independence number.
        
        # Let's implement a standard greedy coloring that returns the number of colors used.
        # To make it fast, we'll just iterate.
        
        # Optimization: If the number of vertices is small, this is fast.
        # For 205, it's fine.
        
        # We can also use the bound: min(|V|, max_degree + 1) is weak.
        # Let's stick to greedy coloring count.
        
        # Better: Use a list of available colors for each vertex?
        # Simpler: 
        # 1. Sort vertices by degree descending.
        # 2. For each vertex, find the smallest color not used by its neighbors.
        
        # To speed up, we can pre-compute neighbor lists for the subgraph?
        # Since candidates change, we compute on the fly.
        
        # Let's try a different upper bound: 
        # The size of the maximum independent set is <= the number of colors in a greedy coloring.
        # We can also use: |V| - min_vertex_cover_size.
        
        # Let's implement the greedy coloring count.
        
        # To make it faster, we can use bitmasks for colors?
        # Since max colors <= 205, we can use a list of masks for each color.
        
        color_masks = []
        
        for deg, v in degrees:
            # Check colors
            assigned = False
            for c_idx, c_mask in enumerate(color_masks):
                # If v is not in c_mask and no neighbor of v is in c_mask
                if not (c_mask & (1 << v)) and not (conflict_mask[v] & c_mask):
                    c_mask |= (1 << v)
                    color_masks[c_idx] = c_mask
                    assigned = True
                    break
            
            if not assigned:
                color_masks.append(1 << v)
                
        return len(color_masks)

    def branch_and_bound(candidates_mask, selected_mask, current_size):
        nonlocal best_size, best_solution_mask
        
        if time.time() > deadline:
            return
            
        if not candidates_mask:
            if current_size > best_size:
                best_size = current_size
                best_solution_mask = selected_mask
            return
            
        # Upper bound using greedy coloring
        # If current_size + upper_bound <= best_size, prune
        ub = greedy_coloring(candidates_mask)
        if current_size + ub <= best_size:
            return
            
        # Choose a vertex to branch on
        # Heuristic: Choose vertex with highest degree in candidate subgraph?
        # Or lowest degree? 
        # Usually, branching on a vertex with high degree reduces the candidate set more.
        
        # Let's pick a vertex. 
        # To improve performance, we can sort vertices in candidates by degree and pick the first.
        
        verts = []
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            verts.append(v)
            cm ^= lsb
            
        if not verts:
            return

        # Calculate degrees in subgraph
        degs = []
        for v in verts:
            neigh = conflict_mask[v] & candidates_mask
            deg = bin(neigh).count('1')
            degs.append((deg, v))
            
        # Pick vertex with highest degree (or lowest? High degree usually prunes more)
        degs.sort(reverse=True)
        v = degs[0][1]
        
        # Branch 1: Include v
        # New selected: selected_mask | (1 << v)
        # New candidates: candidates_mask & ~(1 << v) & ~conflict_mask[v]
        new_candidates_1 = candidates_mask & ~(1 << v) & ~conflict_mask[v]
        branch_and_bound(new_candidates_1, selected_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        # New selected: selected_mask
        # New candidates: candidates_mask & ~(1 << v)
        new_candidates_2 = candidates_mask & ~(1 << v)
        branch_and_bound(new_candidates_2, selected_mask, current_size)

    # Main search loop with random restarts
    # We use random restarts to find good initial lower bounds (best_size)
    # which helps prune the B&B tree.
    
    random.seed(42)
    
    # Initial greedy to set a baseline
    # We can do a few greedy passes to get a decent best_size
    for _ in range(100):
        order = list(range(m))
        random.shuffle(order)
        sel_mask = 0
        sel_list = []
        for v in order:
            if (conflict_mask[v] & sel_mask) == 0:
                sel_mask |= (1 << v)
                sel_list.append(v)
        if len(sel_list) > best_size:
            best_size = len(sel_list)
            best_solution_mask = sel_mask

    # Now run B&B with different orderings or just on the full set
    # Since the graph is fixed, B&B is deterministic given the vertex ordering heuristics.
    # However, the heuristics (greedy coloring, vertex selection) can be varied.
    # For now, we run B&B on the full set of vertices.
    # To allow multiple attempts, we can shuffle the initial vertex list used for degree sorting?
    # The degree is fixed. The tie-breaking in sorting can be randomized.
    
    # Let's run B&B once with a randomized tie-breaker in vertex selection.
    # Or just run it. If it's too slow, the deadline will stop it.
    
    # We can run multiple B&B instances with slightly different heuristics.
    # For simplicity, let's run one robust B&B.
    
    # To make it non-deterministic and explore different parts of the space,
    # we can randomize the order of vertices in the 'degs' list when degrees are equal.
    
    def branch_and_bound_randomized(candidates_mask, selected_mask, current_size):
        nonlocal best_size, best_solution_mask
        
        if time.time() > deadline:
            return
            
        if not candidates_mask:
            if current_size > best_size:
                best_size = current_size
                best_solution_mask = selected_mask
            return
            
        ub = greedy_coloring(candidates_mask)
        if current_size + ub <= best_size:
            return
            
        verts = []
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            verts.append(v)
            cm ^= lsb
            
        if not verts:
            return

        degs = []
        for v in verts:
            neigh = conflict_mask[v] & candidates_mask
            deg = bin(neigh).count('1')
            degs.append((deg, v))
            
        # Randomize ties in degree to explore different branches
        # We can sort by degree, but for equal degrees, randomize order.
        # Python sort is stable, so we can shuffle items with same degree?
        # Easier: Just pick a random vertex from those with max degree?
        # Or sort by (-degree, random.random())
        
        random_degs = [(-deg, random.random(), v) for deg, v in degs]
        random_degs.sort()
        v = random_degs[0][2]
        
        # Branch 1: Include v
        new_candidates_1 = candidates_mask & ~(1 << v) & ~conflict_mask[v]
        branch_and_bound_randomized(new_candidates_1, selected_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        new_candidates_2 = candidates_mask & ~(1 << v)
        branch_and_bound_randomized(new_candidates_2, selected_mask, current_size)

    # Run the randomized B&B
    # We might run it multiple times if time permits, but B&B is usually deep.
    # Let's run it once with a generous time limit.
    
    full_mask = (1 << m) - 1
    branch_and_bound_randomized(full_mask, 0, 0)
    
    # If time remains, we can try to improve with local search on the best found
    if time.time() < deadline - 10 and best_solution_mask:
        # Convert mask to list
        best_list = []
        mm = best_solution_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            best_list.append(v)
            mm ^= lsb
            
        # Try to add any vertex that fits? (Shouldn't be possible if B&B was exhaustive for that size, 
        # but B&B stops at deadline, so it might not be maximal)
        # Greedy add
        improved = True
        while improved:
            improved = False
            for v in range(m):
                if not (best_solution_mask & (1 << v)):
                    if (conflict_mask[v] & best_solution_mask) == 0:
                        best_solution_mask |= (1 << v)
                        best_size += 1
                        improved = True
                        break

    # Extract final solution
    if best_solution_mask == 0:
        # Fallback if B&B failed or returned empty
        # Use the best greedy result
        best_list = []
        mm = best_solution_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            best_list.append(v)
            mm ^= lsb
    else:
        best_list = []
        mm = best_solution_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            best_list.append(v)
            mm ^= lsb

    return np.array(best_list, dtype=np.int64)
