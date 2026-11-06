"""Optimized construction for large pairwise noncoprime sets containing 255."""
import math
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Find largest subset of [1,255] containing 255 with gcd(a,b)>1 for all distinct pairs.
    
    # Generate all numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency using bitsets
    # adj[i] is a bitmask where bit j is 1 if candidates[i] and candidates[j] are coprime-compatible (gcd > 1)
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
    # Precompute degrees for sorting
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree (descending) to improve coloring bound
    # We need to map old indices to new indices
    sorted_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    # Create mapping: new_index -> old_index
    new_to_old = sorted_indices
    # Create mapping: old_index -> new_index
    old_to_new = [0] * n
    for new_i, old_i in enumerate(new_to_old):
        old_to_new[old_i] = new_i
    
    # Rebuild adjacency in sorted order
    new_adj = [0] * n
    for new_i in range(n):
        old_i = new_to_old[new_i]
        mask = adj[old_i]
        new_mask = 0
        # Convert old mask to new mask
        # Instead of iterating all bits, we can just check each candidate neighbor
        # But n is small (~100), so simple loop is fine
        for new_j in range(n):
            if new_j == new_i: continue
            old_j = new_to_old[new_j]
            if (adj[old_i] >> old_j) & 1:
                new_mask |= (1 << new_j)
        new_adj[new_i] = new_mask
        
    # Find index of 255 in sorted order
    target_idx = old_to_new[idx_255]
    
    # Branch and Bound for Maximum Clique
    # We must include target_idx
    # Initialize best clique with target_idx
    best_clique = [target_idx]
    best_size = 1
    
    # Candidates are all other vertices
    # We can restrict candidates to those connected to target_idx
    # Since we want a clique containing target_idx, all other vertices must be connected to target_idx
    # And to each other.
    
    # Let's run B&B on the subgraph induced by neighbors of target_idx + target_idx
    # Actually, standard B&B handles this. We just force target_idx to be in the first layer.
    
    def expand(candidates_mask, current_clique, current_size):
        """
        candidates_mask: bitmask of vertices that can be added to current_clique
        current_clique: list of vertices in the current clique
        current_size: len(current_clique)
        """
        nonlocal best_clique, best_size
        
        if not candidates_mask:
            return
        
        # Get list of vertices in candidates_mask
        # Order them by degree in the candidate subgraph? 
        # Standard B&B uses a coloring bound.
        
        # Greedy Coloring on the candidate subgraph
        # We assign colors to vertices in candidates_mask.
        # The number of colors is an upper bound on the clique size in this subgraph.
        
        # To make coloring effective, we should process vertices in an order that creates a good bound.
        # Usually, sorting by degree in the candidate subgraph helps.
        
        # Extract vertices
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            verts.append(idx)
            m ^= lsb
            
        # If we have a time check here, it's expensive. 
        # n is small, so we can do it.
        
        # Sort vertices by degree in the candidate subgraph (descending)
        # Degree in candidate subgraph for v is popcount(adj[v] & candidates_mask)
        def deg_in_sub(v):
            return bin(new_adj[v] & candidates_mask).count('1')
        
        verts.sort(key=deg_in_sub, reverse=True)
        
        # Greedy Coloring
        # color[v] = color assigned to v
        # We want to partition verts into independent sets in the complement graph?
        # No, in the Clique problem, we color the *candidate graph*.
        # An independent set in the candidate graph is a clique in the complement?
        # Wait. 
        # The bound is: Max Clique in G <= Number of Colors in a proper coloring of G.
        # So we color the graph G (where edges are "gcd>1").
        # A proper coloring means no two adjacent vertices have the same color.
        # So a color class is an independent set in G.
        # The size of the largest independent set is not the bound.
        # The bound is the *number* of colors used.
        
        colors = {}
        color_classes = [] # list of sets (bitmasks) for each color
        current_color = 0
        uncolored = set(verts)
        
        # To get a tight bound, we should pick vertices that have many connections to uncolored vertices?
        # Or just iterate through the sorted list.
        
        # Standard Tomita algorithm approach:
        # Iterate through vertices. Assign smallest possible color.
        
        # For each vertex v in verts (sorted by degree):
        for v in verts:
            # Find a color for v
            # v cannot share a color with any of its neighbors in the candidate set that are already colored
            # Actually, we are coloring the subgraph induced by `verts`.
            # Neighbors of v in this subgraph are `new_adj[v] & candidates_mask`.
            
            # We maintain a list of color classes.
            # If v is not adjacent to any vertex in color_class[c], we can assign c.
            
            assigned = False
            for c in range(len(color_classes)):
                # Check if v is adjacent to any vertex in color_classes[c]
                # color_classes[c] is a bitmask
                # If (new_adj[v] & color_classes[c]) == 0, then v is not adjacent to any in this class
                # So they form an independent set.
                if (new_adj[v] & color_classes[c]) == 0:
                    color_classes[c] |= (1 << v)
                    colors[v] = c
                    assigned = True
                    break
            
            if not assigned:
                # New color
                color_classes.append(1 << v)
                colors[v] = len(color_classes) - 1
                current_color += 1
                
        # Upper bound: current_size + len(color_classes)
        if current_size + len(color_classes) <= best_size:
            return
            
        # Branching
        # We should branch on the vertices in the order of their colors?
        # Tomita: Branch on vertices in the order they were colored?
        # Usually, we branch on the last color class, or the first?
        # To find a large clique quickly, we often branch on high-degree vertices.
        # But the coloring bound prunes based on the total number of colors.
        
        # Let's branch on vertices in `verts` order (which is degree-sorted).
        # However, we must ensure we don't miss the optimal solution.
        # Standard B&B:
        # For each vertex v in the candidate set (in some order):
        #   Add v to clique
        #   Recurse on neighbors of v in the candidate set (that come after v in the order? No, all neighbors)
        #   But we must maintain the property that we are building a clique.
        #   The new candidate set is (candidates_mask & new_adj[v]).
        #   To avoid duplicates and ensure we explore all, we typically remove v and its non-neighbors from the candidate set for the next level?
        #   Actually, if we pick v, the next vertices must be neighbors of v.
        #   So new_candidates = candidates_mask & new_adj[v].
        #   We also remove v from candidates_mask for the "skip v" branch?
        
        # Efficient implementation:
        # We iterate through vertices in `verts`.
        # If we pick v, we recurse on (candidates_mask & new_adj[v]) with v added.
        # If we skip v, we remove v from candidates_mask and continue.
        # But the coloring bound was computed on the *current* candidates_mask.
        # If we skip v, the bound decreases.
        
        # A common optimization is to only branch on vertices that are "promising".
        
        # Let's use the standard recursive structure:
        # expand(candidates, clique)
        # 1. Color candidates.
        # 2. If |clique| + num_colors <= best, return.
        # 3. For each v in candidates (ordered by color class, last to first? or first to last?):
        #    If |clique| + color[v] <= best, break? (If we process in reverse color order)
        #    Pick v.
        #    Recurse on (candidates & neighbors(v)).
        #    Remove v from candidates.
        #    Update bound?
        
        # Let's try a simpler, robust B&B.
        
        # We will iterate through `verts`.
        # We maintain a local candidate mask `cand_mask` which starts as `candidates_mask`.
        # We also maintain the coloring for the *remaining* candidates?
        # This is complex to update incrementally.
        
        # Given n ~ 100, we can afford to re-color at each node if the depth is small.
        # But B&B can go deep.
        
        # Alternative: Use the coloring bound to order the branching.
        # Vertices in the last color class are the "most constrained" in some sense?
        # Actually, Tomita's MCQ algorithm orders vertices by color.
        # It processes vertices in reverse color order.
        # If a vertex has color c, and current_size + c <= best_size, we can stop branching on this vertex and subsequent ones (if they have lower/equal color)?
        # Wait, if we process in reverse color order (highest color first), and we find that current_size + color[v] <= best, then any vertex with color <= color[v] can't improve the best.
        
        # So:
        # 1. Color vertices.
        # 2. Sort vertices by color (ascending).
        # 3. Iterate from last vertex (highest color) down to first.
        # 4. If current_size + color[v] <= best_size, break.
        # 5. Branch on v:
        #    new_clique = current_clique + [v]
        #    new_candidates = candidates_mask & new_adj[v]
        #    expand(new_candidates, new_clique, current_size + 1)
        # 6. Remove v from candidates_mask (for the "skip" path, which is implicit in the loop moving to the next vertex).
        #    BUT, the coloring was done on the original candidates_mask.
        #    If we remove v, the colors of remaining vertices might change (could decrease).
        #    So the bound `color[v]` might be too high for the remaining set?
        #    No, the bound is valid for the set *before* removal.
        #    If we skip v, we are left with a subset. The clique size in the subset is <= clique size in the original set.
        #    So the bound holds.
        
        # Implementation details:
        # We need to map `verts` to their colors.
        
        # Re-sort verts by color
        # We need a list of (vertex, color) pairs
        # And we need to ensure that when we branch, we only consider vertices that are still in `candidates_mask`.
        
        # Let's refine the loop:
        # We have `verts` sorted by color.
        # We iterate `i` from len(verts)-1 down to 0.
        # v = verts[i]
        # if current_size + colors[v] <= best_size: break
        # 
        # # Branch on v
        # if v in candidates_mask: # It should be, unless removed by previous branches? 
        #   # In this loop structure, we are effectively doing "skip previous vertices".
        #   # So v is always in the candidate set at the start of the iteration for it.
        #   new_clique = current_clique + [v]
        #   new_candidates = candidates_mask & new_adj[v]
        #   # Note: new_candidates should not include v itself? 
        #   # new_adj[v] does not include v (no self loops).
        #   expand(new_candidates, new_clique, current_size + 1)
        #   
        #   # Remove v from candidates_mask for subsequent iterations (which represent skipping v)
        #   candidates_mask &= ~(1 << v)
        #   
        #   # Optimization: If we removed v, do we need to update colors of remaining?
        #   # No, because the bound `colors[u]` for remaining u is an upper bound on the clique in the *original* set.
        #   # The clique in the reduced set is <= clique in original set.
        #   # So the bound is still valid (though potentially loose).
        
        # This seems correct and efficient.
        
        # One issue: `colors` dict was built for `verts`.
        # We need to ensure `verts` is sorted by color.
        
        # Let's implement this inside expand.
        
        # We already computed `color_classes` and `colors`.
        # We need `verts` sorted by `colors[v]`.
        
        # Note: The coloring step above assigned colors greedily.
        # We should ensure `verts` is sorted by color for the branching loop.
        
        # Let's re-sort `verts` by color
        # We need to handle ties in color consistently.
        
        # Actually, the standard Tomita algorithm sorts by color, and for vertices with the same color, it doesn't matter much, but degree is a good tie-breaker.
        
        # Let's re-sort `verts` by (color, degree)
        # Wait, `verts` was sorted by degree initially for coloring.
        # Now we sort by color.
        
        # We need to be careful: `colors` is a dict {v: c}.
        
        # Let's create a list of (v, c) and sort it.
        
        # However, we must only consider vertices that are in `candidates_mask`.
        # In the loop `for i in range(len(verts)-1, -1, -1):`, `verts[i]` is always in `candidates_mask` at the start of the iteration?
        # Yes, because we only remove vertices *after* branching on them.
        
        # Let's code this.
        
        # We need `verts` sorted by color.
        # We also need to know the color of each vertex.
        
        # The `colors` dict from the greedy coloring step is valid for the current `candidates_mask`.
        
        # Sort verts by color
        # We need to access colors[v]
        
        # One detail: The greedy coloring I wrote above iterates through `verts` (sorted by degree).
        # This produces a specific coloring.
        
        # Let's proceed.
        
        # Optimization: If `candidates_mask` is small, we can just brute force?
        # No, B&B handles it.
        
        # Let's add a time check.
        if time.time() > deadline:
            return
            
        # Sort vertices by color
        # We need a stable sort or just sort by color.
        # Since we want to break early if `current_size + color[v] <= best_size`,
        # and we iterate from highest color to lowest,
        # we should sort `verts` by color ascending, and iterate backwards.
        
        # But `verts` is a list of indices.
        # Let's create a sorted list of indices based on their assigned color.
        
        # Note: The coloring is done on the subgraph induced by `verts`.
        
        # Let's sort `verts` by `colors[v]`
        # We need to handle the case where `colors` might not have an entry? 
        # No, every v in `verts` got a color.
        
        # To make the bound tighter, we can re-color after removing vertices?
        # That's expensive. The standard approach is to keep the bound from the entry of the function.
        
        # Let's sort
        # We need to sort `verts` by color.
        # `colors` is a dict.
        
        # However, sorting a list of 100 items is fast.
        
        # One potential issue: The greedy coloring depends on the order of `verts`.
        # I sorted `verts` by degree before coloring.
        # This is a good heuristic.
        
        # Let's refine the coloring to be more aggressive?
        # No, standard greedy is fine.
        
        # Let's implement the branching.
        
        # We need to update `best_clique` and `best_size` if we find a better one.
        
        # Note: `current_clique` is a list.
        
        # We must ensure we don't modify `current_clique` in a way that affects the parent.
        # We pass a new list `current_clique + [v]`.
        
        # Let's run this.
        
        # To speed up, we can check if `current_size + len(color_classes) <= best_size` before branching.
        # We already did that.
        
        # Branching loop
        # We need `verts` sorted by color.
        
        # Let's create a list of (color, vertex) pairs
        # And sort by color.
        
        # Actually, we can just sort `verts` using `colors` as key.
        
        # But `colors` is a dict.
        
        # Let's do it.
        
        # Wait, I should check if `v` is still in `candidates_mask`?
        # In the loop `for i ...`, we remove `verts[i]` from `candidates_mask` after branching.
        # So for `i-1`, `verts[i-1]` is still in `candidates_mask`.
        # So it's safe.
        
        # Let's write the code.
        
        # One more thing: `new_adj[v]` includes v's neighbors.
        # `new_candidates = candidates_mask & new_adj[v]`
        # This correctly restricts the next level to neighbors of v that are in the current candidate set.
        
        # Let's execute.
        
        # I will put the B&B logic in a separate function for clarity?
        # No, it's nested in expand.
        
        # Let's optimize the coloring loop.
        # Instead of `for v in verts`, which is sorted by degree,
        # we can just use that order.
        
        # The colors dict is ready.
        
        # Sort verts by color
        # We need to be careful with the sort key.
        
        # Let's try.
        
        # To avoid sorting overhead, we can just iterate `verts` and check the bound?
        # No, we need the bound to break early.
        # So we MUST sort by color.
        
        # Let's sort.
        
        # Note: `verts` is a list of ints.
        
        # We can sort `verts` in place?
        # `verts.sort(key=lambda v: colors[v])`
        
        # But we need to iterate from end to start.
        
        # Let's do it.
        
        # Also, we should update `best_size` if `current_size > best_size`?
        # No, `current_size` is the size of the *current* clique, which is a valid clique.
        # So yes, we should update best if we find a larger one.
        # But the B&B logic usually updates best when it reaches a leaf?
        # Or whenever `current_size` increases?
        # Yes, any valid clique is a candidate for best.
        
        # So at the start of `expand`, if `current_size > best_size`, update.
        
        # Let's add that.
        
        if current_size > best_size:
            best_size = current_size
            best_clique = current_clique[:] # Copy
        
        # Now the branching.
        
        # Sort verts by color
        # We need to ensure `colors` is consistent.
        
        # Let's sort.
        verts.sort(key=lambda v: colors[v])
        
        # Iterate backwards
        for i in range(len(verts) - 1, -1, -1):
            v = verts[i]
            
            # Bound check
            if current_size + colors[v] <= best_size:
                break
                
            # Branch on v
            # Check if v is in candidates_mask (it should be)
            if not (candidates_mask & (1 << v)):
                continue # Should not happen, but safe
                
            # New clique
            new_clique = current_clique + [v]
            
            # New candidates
            # Neighbors of v in the current candidate set
            new_candidates = candidates_mask & new_adj[v]
            
            # Recurse
            expand(new_candidates, new_clique, current_size + 1)
            
            # Remove v from candidates_mask for the "skip" branch
            candidates_mask &= ~(1 << v)
            
            # Time check
            if time.time() > deadline:
                return
                
    # Start B&B
    # Initial candidates: all vertices except target_idx?
    # No, we start with `current_clique = [target_idx]`.
    # Candidates: all vertices that are neighbors of target_idx.
    # Because any vertex in the clique must be connected to target_idx.
    
    initial_candidates = new_adj[target_idx]
    # Remove target_idx from candidates?
    # new_adj[target_idx] does not contain target_idx (no self loops).
    # So it's clean.
    
    # Call expand
    expand(initial_candidates, [target_idx], 1)
    
    # Convert best_clique (indices in sorted order) back to original values
    # best_clique contains indices in `new_to_old` mapping?
    # No, `new_adj` and `target_idx` are in the *sorted* index space.
    # So `best_clique` contains sorted indices.
    # We need to map them back to original values.
    
    result_indices = [new_to_old[i] for i in best_clique]
    result = [candidates[i] for i in result_indices]
    result.sort()
    
    # Verify
    assert 255 in result
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            assert math.gcd(result[i], result[j]) > 1
            
    return result

# EVOLVE-BLOCK-END
