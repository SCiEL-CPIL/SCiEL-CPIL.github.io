"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Precompute all valid subsets (cliques or independent sets)
    # A subset S is valid if it is a clique or an independent set.
    # For n=14, 2^14 = 16384 subsets. This is small.
    valid_masks = set()
    # We represent subsets as bitmasks.
    for mask in range(1, 1 << n):
        # Check if clique
        is_clique = True
        # Check if independent set
        is_indep = True
        
        # Get vertices in mask
        verts = [i for i in range(n) if (mask >> i) & 1]
        
        if len(verts) > 1:
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if adj[u][v]:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
        
        if is_clique or is_indep:
            valid_masks.add(mask)
            
    # Backtracking search for minimum k
    # We want to partition vertices 0..n-1 into k parts.
    # Each part must correspond to a valid mask.
    # The union of parts must be all vertices, and parts must be disjoint.
    
    # Sort vertices by degree to help pruning? Or just use bitmask DP/backtracking.
    # Let's use a recursive function that assigns vertices to parts.
    # To avoid symmetry, we can enforce an ordering on the parts.
    # However, since parts are indistinguishable except for content, we can just pick the next unassigned vertex and try to put it in an existing part or start a new one.
    
    # A better approach for small n:
    # Iterate k from 1 to n.
    # Check if there is a partition into k valid sets.
    # This is equivalent to covering the vertex set with k valid sets (disjoint).
    
    # Let's use a simple backtracking on the "next vertex to assign".
    # State: current_mask (assigned vertices), list of current_parts_masks.
    # To reduce symmetry, we can require that the masks in parts are in non-increasing order of their lowest set bit? No, that's complex.
    # Simpler: assign vertices 0, 1, ..., n-1 one by one.
    # For vertex v, try adding it to each existing part if the resulting set is still valid (clique or indep).
    # Also try starting a new part with {v}.
    # To break symmetry:
    # 1. If we start a new part, its mask should be "smaller" or "larger" than previous new parts?
    # 2. Standard symmetry breaking: The set of parts is a multiset. We can enforce that the parts are sorted by their minimum vertex.
    #    Since we assign vertices in increasing order, the first vertex in a part is the smallest.
    #    So, when we start a new part with vertex v, it must be the case that no previous part has a minimum vertex greater than v? No.
    #    Actually, since we process v in order 0..n-1, any part containing v will have min vertex <= v.
    #    If we start a new part with v, its min is v.
    #    Any existing part has min < v (since it was created earlier with a smaller vertex).
    #    So this naturally handles some symmetry.
    #    Another symmetry: swapping two parts that are both "independent" or both "clique" doesn't change validity.
    #    We can enforce that the list of part masks is sorted lexicographically or by value.
    
    # Let's try a DP or simple backtracking with memoization on (assigned_mask, tuple(sorted(parts_masks))).
    # But parts_masks can be many.
    
    # Given n=14, a simple backtracking with pruning should be fast enough.
    # Pruning:
    # 1. If current number of parts + remaining vertices > k, fail. (Not very tight)
    # 2. If a remaining vertex cannot be added to any existing part and we can't start a new one (if we hit limit k), fail.
    
    # Let's implement a recursive search for a fixed k.
    # If found, return the partition.
    
    def is_valid_part(mask):
        return mask in valid_masks

    def search(k, assigned_mask, parts_masks):
        # parts_masks is a tuple of masks, sorted for symmetry breaking
        if assigned_mask == (1 << n) - 1:
            return True if len(parts_masks) <= k else False
        
        # Find next unassigned vertex
        next_v = -1
        for v in range(n):
            if not (assigned_mask >> v) & 1:
                next_v = v
                break
        
        if next_v == -1:
            return len(parts_masks) <= k

        # Option 1: Add next_v to an existing part
        # To break symmetry, we only try adding to parts that are "compatible" in some ordering?
        # Actually, just try all. If two parts are identical, we only need to try one.
        # We can track unique part masks.
        tried_masks = set()
        
        for i, pm in enumerate(parts_masks):
            if pm in tried_masks: continue
            tried_masks.add(pm)
            
            new_pm = pm | (1 << next_v)
            if is_valid_part(new_pm):
                # Update parts_masks: replace pm with new_pm, then sort
                new_parts_list = list(parts_masks)
                new_parts_list[i] = new_pm
                new_parts_tuple = tuple(sorted(new_parts_list))
                
                if search(k, assigned_mask | (1 << next_v), new_parts_tuple):
                    return True
        
        # Option 2: Start a new part with next_v
        if len(parts_masks) < k:
            new_pm = 1 << next_v
            # Symmetry breaking:
            # If we start a new part, and there are other parts with the same "type" or value, we might be duplicating work.
            # Since we sort parts_masks, starting a new part with mask m adds m to the tuple.
            # We should ensure we don't create a configuration that is a permutation of a previously tried one.
            # The sorted tuple handles this.
            
            new_parts_list = list(parts_masks)
            new_parts_list.append(new_pm)
            new_parts_tuple = tuple(sorted(new_parts_list))
            
            if search(k, assigned_mask | (1 << next_v), new_parts_tuple):
                return True
        
        return False

    # We need to recover the actual partition, not just True/False.
    # Let's modify search to return the parts if successful.
    
    def search_with_solution(k, assigned_mask, parts_masks):
        if assigned_mask == (1 << n) - 1:
            if len(parts_masks) <= k:
                return [list(mask_to_verts(m)) for m in parts_masks]
            else:
                return None
        
        next_v = -1
        for v in range(n):
            if not (assigned_mask >> v) & 1:
                next_v = v
                break
        
        if next_v == -1:
            if len(parts_masks) <= k:
                return [list(mask_to_verts(m)) for m in parts_masks]
            else:
                return None

        tried_masks = set()
        
        for i, pm in enumerate(parts_masks):
            if pm in tried_masks: continue
            tried_masks.add(pm)
            
            new_pm = pm | (1 << next_v)
            if is_valid_part(new_pm):
                new_parts_list = list(parts_masks)
                new_parts_list[i] = new_pm
                new_parts_tuple = tuple(sorted(new_parts_list))
                
                res = search_with_solution(k, assigned_mask | (1 << next_v), new_parts_tuple)
                if res is not None:
                    return res
        
        if len(parts_masks) < k:
            new_pm = 1 << next_v
            new_parts_list = list(parts_masks)
            new_parts_list.append(new_pm)
            new_parts_tuple = tuple(sorted(new_parts_list))
            
            res = search_with_solution(k, assigned_mask | (1 << next_v), new_parts_tuple)
            if res is not None:
                return res
        
        return None

    def mask_to_verts(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    # Fallback greedy solution
    def greedy_solution():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                # Check if trial is clique or independent
                is_clique = True
                is_indep = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        if adj[trial[i]][trial[j]]:
                            is_indep = False
                        else:
                            is_clique = False
                        if not is_clique and not is_indep:
                            break
                    if not is_clique and not is_indep:
                        break
                if is_clique or is_indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best_solution = greedy_solution()
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        res = search_with_solution(k, 0, ())
        if res is not None:
            # Verify solution
            # Check all vertices present exactly once
            all_verts = []
            for part in res:
                all_verts.extend(part)
            if sorted(all_verts) != list(range(n)):
                continue
            # Check each part is valid
            valid = True
            for part in res:
                mask = 0
                for v in part:
                    mask |= (1 << v)
                if mask not in valid_masks:
                    valid = False
                    break
            if valid:
                return res
        # If k=1 fails, try k=2, etc.
        
    return best_solution
# EVOLVE-BLOCK-END
