"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
    
    # Precompute adjacency as bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute: for each subset (bitmask), is it a clique? is it an independent set?
    # We will store these in lists for O(1) lookup.
    # is_clique[mask] = 1 if mask is a clique
    # is_indep[mask] = 1 if mask is an independent set
    
    is_clique = [0] * (1 << n)
    is_indep = [0] * (1 << n)
    
    is_clique[0] = 1
    is_indep[0] = 1
    
    # Efficiently compute properties
    # For each mask, pick a vertex v in mask, rest = mask ^ (1<<v)
    # mask is clique iff rest is clique AND v is connected to all in rest
    # mask is indep iff rest is indep AND v is NOT connected to any in rest
    
    for mask in range(1, 1 << n):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        if is_clique[rest]:
            if (adj[v] & rest) == rest:
                is_clique[mask] = 1
        
        if is_indep[rest]:
            if (adj[v] & rest) == 0:
                is_indep[mask] = 1

    full_mask = (1 << n) - 1
    
    # Helper to get list of vertices from mask
    def mask_to_list(mask):
        return [i for i in range(n) if mask & (1 << i)]

    # IDDFS: Try to cover 'remaining' with exactly 'max_depth' parts.
    # Returns a list of masks if successful, else None.
    def solve_iddfs(remaining, max_depth, current_path):
        if remaining == 0:
            return current_path
        
        if max_depth == 0:
            return None
            
        # If remaining is a single part (clique or independent), we can take it all
        # This is a strong pruning step
        if is_clique[remaining] or is_indep[remaining]:
            # We can cover it in 1 part.
            # If max_depth >= 1, we can just take this one part and we are done?
            # No, we must use exactly max_depth? No, we want min parts.
            # IDDFS checks if it's possible in <= max_depth.
            # If we can cover in 1, and max_depth >= 1, we found a solution.
            # But we need to return a valid partition of size <= max_depth.
            # Actually, if we find a cover of size k, we can pad it? No, parts must be non-empty.
            # So we need to find a cover of size exactly max_depth? 
            # No, standard IDDFS: "Can we solve in depth D?"
            # If yes, return the solution.
            # If we can cover 'remaining' in 1 part, and max_depth >= 1, 
            # we can just return [remaining] + [dummy parts]? No, dummy parts must be non-empty.
            # So if we have 'remaining' and we can cover it in 1 part, we have used 1 part.
            # If max_depth > 1, we still have "unused" depth. But the problem asks for MIN parts.
            # So if we find a solution in depth D, that IS the minimum if we iterate D=1, 2, 3...
            # So if we succeed at depth D, we return a partition of size D.
            # If 'remaining' is a valid part, we can just assign it to the last slot?
            # Wait, if I have max_depth=3, and I cover remaining in 1 part, I have a valid partition of size 1.
            # Since I'm checking D=1 first, I would have found it.
            # So in the recursive step, if I am at depth D, I should try to split 'remaining' into 1 part + (D-1) parts.
            
            # Optimization: If 'remaining' is a valid part, and max_depth == 1, success.
            # If max_depth > 1, we don't want to just take it all if we can split it?
            # No, taking it all is better (fewer parts). But we are constrained by max_depth?
            # No, IDDFS asks "is there a solution of size <= max_depth?".
            # If I can do it in 1, and max_depth is 5, I can do it in 5? No, I can't add empty parts.
            # So "Can we solve in <= max_depth" is equivalent to "Can we solve in exactly max_depth" 
            # ONLY IF we can split parts. But we can't split a valid part into two valid parts necessarily.
            # Example: Clique of 3. Can be 1 part. Can it be 2 parts? Yes, {1,2} and {3}.
            # So we can always split a part into smaller valid parts?
            # A clique can be split into smaller cliques.
            # An independent set can be split into smaller independent sets.
            # So if a solution of size K exists, a solution of size K+1 exists (by splitting one part of size >= 2).
            # Therefore, finding a solution of size <= max_depth is the same as finding one of size exactly max_depth 
            # (provided we can always split, which we can for non-singleton parts).
            # However, singletons cannot be split. So if the solution is all singletons, we can't increase depth.
            # But we are looking for MIN depth. So we iterate D=1, 2, 3...
            # If we find a solution for D, it is the minimum.
            
            if max_depth == 1:
                return current_path + [remaining]
            
            # If max_depth > 1, we MUST split 'remaining' into at least 2 parts.
            # So we cannot just take 'remaining' as the whole part if max_depth > 1, 
            # UNLESS we are allowed to have fewer parts? 
            # The IDDFS logic: "Is there a partition of 'remaining' into at most max_depth parts?"
            # If I can do it in 1, the answer is Yes.
            # So I should return the partition of size 1.
            # But the caller expects to use up the depth? No, the caller just wants a valid partition.
            # So if I return [remaining], the total depth used is 1.
            # This is valid for "depth <= max_depth".
            
            return current_path + [remaining]

        # Choose the most constrained vertex in 'remaining'
        # Heuristic: Vertex with fewest valid subsets containing it within 'remaining'
        # Or simply pick the first one to keep it simple/fast, but "most constrained" is better.
        # Let's pick the vertex with the smallest number of "candidate parts" it can belong to.
        
        # To speed up, we can just pick the lowest vertex, as n is small.
        # But let's try a slightly better heuristic: pick the vertex v in remaining
        # such that the number of valid subsets S (subset of remaining, containing v) is minimized.
        
        # For n=14, iterating all subsets is expensive if done naively.
        # Let's just pick the lowest set bit. It's fast and effective enough for n=14.
        lb = remaining & (-remaining)
        v = lb.bit_length() - 1
        
        # Generate all valid subsets of 'remaining' that contain 'v'
        # We can iterate submasks of 'remaining' that contain 'v'
        # sub = remaining
        # while sub:
        #   if sub has v and (is_clique[sub] or is_indep[sub]):
        #       ...
        #   sub = (sub-1) & remaining
        
        # To make it faster, we can precompute for each vertex v and mask M, the list of valid parts?
        # That's too much memory.
        
        # Let's iterate submasks.
        sub = remaining
        candidates = []
        while sub:
            if (sub & lb): # Contains v
                if is_clique[sub] or is_indep[sub]:
                    candidates.append(sub)
            sub = (sub - 1) & remaining
            
        # Sort candidates by size? Larger parts are better (greedy).
        # Or smaller? 
        # Usually, trying larger valid parts first reduces the remaining set faster.
        candidates.sort(key=lambda x: -x.bit_count())
        
        for part in candidates:
            new_remaining = remaining ^ part
            # Pruning: if new_remaining can be covered in max_depth - 1 parts?
            # We just recurse.
            res = solve_iddfs(new_remaining, max_depth - 1, current_path + [part])
            if res:
                return res
        return None

    # Iterative Deepening
    for depth in range(1, n + 1):
        res = solve_iddfs(full_mask, depth, [])
        if res:
            # res is a list of masks
            partition = [mask_to_list(m) for m in res]
            # Ensure types are Python int
            final_result = []
            for part in partition:
                final_result.append([int(v) for v in part])
            return final_result

    # Fallback (should not happen)
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
