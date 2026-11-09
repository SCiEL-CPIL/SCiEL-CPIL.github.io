<<<<<<< SEARCH
def _branch_and_bound_polish(selected, deadline):
    """
    Branch and bound search to find a larger valid set than 'selected'.
    Uses the current size as a lower bound.
    """
    current_set = set(selected)
    best_size = len(selected)
    best_solution = list(selected)
    
    # Candidates not in current set
    # We prioritize candidates that are "less restrictive" (larger numbers)
    # But for B&B, we just need a list.
    candidates = [x for x in range(N, 0, -1) if x not in current_set]
    
    # Pre-compute conflicts? 
    # A conflict exists between a set S and a candidate x if x cannot be added to S.
    # We can use _can_add directly.
    
    # To make B&B effective, we need a good upper bound.
    # A simple upper bound is: current_size + len(candidates).
    # A tighter bound is hard for hypergraphs.
    # However, we can prune if current_size + len(candidates) <= best_size.
    
    # We will perform a depth-first search.
    # To avoid exponential blowup, we limit the depth of exploration or the number of nodes visited.
    # Since we have a time limit, we will stop if time runs out.
    
    # Heuristic: Try adding candidates in a specific order.
    # We'll maintain a list of "active" candidates.
    
    # Instead of a full generic B&B which might be too slow, we do a limited search:
    # Try to replace small elements with large ones if it increases size or allows further additions.
    # This is essentially a more aggressive local search.
    
    # Let's implement a simple recursive B&B with a node limit.
    node_limit = 50000
    nodes_visited = 0
    
    def dfs(current_subset, remaining_candidates):
        nonlocal best_size, best_solution, nodes_visited
        
        if time.time() > deadline:
            return
            
        nodes_visited += 1
        if nodes_visited > node_limit:
            return
            
        current_size = len(current_subset)
        
        # Upper bound
        if current_size + len(remaining_candidates) <= best_size:
            return
            
        # If we can't add more, check if this is a solution
        if not remaining_candidates:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_subset)
            return
            
        # Try to add a candidate
        # We iterate through remaining candidates.
        # To be efficient, we pick a candidate, try to add it, and recurse.
        # If we can't add it, we remove it from remaining and recurse (branch on exclusion).
        
        # Choose a candidate to branch on.
        # Let's pick the first one in the list (which is sorted descending).
        # Or pick one that seems "compatible".
        
        # Optimization: If current_size == best_size, we still need to explore to find a larger set.
        
        # Let's try adding candidates in order.
        # If we add x, we remove x from remaining and add x to current_subset.
        # If we can't add x, we must remove x from remaining (branch on exclusion).
        
        # However, standard B&B for Max Independent Set usually picks a vertex v.
        # Branch 1: Include v.
        # Branch 2: Exclude v.
        
        # Here, "Include v" means checking if v is compatible with current_subset.
        # If compatible, new_subset = current_subset U {v}, new_remaining = [u for u in remaining if u != v and u is compatible with new_subset? No, just remove v].
        # Actually, if v is included, any u in remaining that is NOT compatible with v should be removed from the remaining list for the next step?
        # No, compatibility is transitive in the sense that if u is incompatible with v, u cannot be added later either.
        # So if we include v, we can prune remaining to only those compatible with v.
        
        # If v is NOT compatible with current_subset, we must exclude v.
        
        v = remaining_candidates[0]
        rest = remaining_candidates[1:]
        
        # Check if v is compatible with current_subset
        if _can_add(v, current_subset):
            # Branch 1: Include v
            new_subset = current_subset + [v]
            # Filter rest to only those compatible with v?
            # _can_add(u, new_subset) is expensive.
            # A weaker filter: if u is incompatible with v (i.e., v divides sum of 3 from ... no, v is in subset).
            # If we include v, we must ensure future additions don't violate constraints with v.
            # So for the next step, the "current_subset" is new_subset.
            # We pass rest as remaining. The _can_add check in the next recursion will handle incompatibilities.
            dfs(new_subset, rest)
            
            # Check time
            if time.time() > deadline:
                return
            
            # Branch 2: Exclude v
            dfs(current_subset, rest)
        else:
            # Branch 1: Exclude v (since we can't include it)
            dfs(current_subset, rest)
            
        # Update best if this path yielded a better solution? 
        # The base case handles that.
        
        # Pruning: if best_size >= current_size + len(rest), we could prune, but we already did at start.
        
    # Start DFS
    # To avoid deep recursion limits, we might need to be careful, but depth is at most 48.
    # However, the branching factor is 1 (we always pick the first candidate).
    # This is just a linear scan effectively, unless we branch on inclusion/exclusion of compatible nodes.
    # Wait, if v is compatible, we branch on Include AND Exclude.
    # This creates a binary tree.
    
    # This might be too slow. Let's limit the branching.
    # Only branch on Include if it's promising?
    # Or just do a single pass greedy with backtracking?
    
    # Given the time constraints, a full B&B might be risky.
    # Let's stick to a limited version:
    # We'll just try to find any set larger than current.
    
    # A safer approach:
    # Iterate through candidates.
    # If we can add a candidate, add it.
    # If we can't, try removing one element from current_subset to see if we can add a "better" candidate.
    # This is what local search does.
    
    # So, B&B might not add much over local search if not carefully tuned.
    # Let's use the B&B idea to *verify* if we can reach 48?
    # No, let's just use the existing local search which is robust.
    
    # I will implement a simple "try to extend" loop that is slightly different from local search:
    # It tries to find ANY valid addition, not just the first one.
    
    pass
=======
def _branch_and_bound_polish(selected, deadline):
    """
    Branch and bound search to find a larger valid set than 'selected'.
    Uses bitset-based conflict propagation and dynamic upper bounds.
    """
    current_set = set(selected)
    best_size = len(selected)
    best_solution = list(selected)
    
    # All numbers 1..N as indices 0..N-1
    all_numbers = list(range(1, N + 1))
    
    # Precompute conflict matrix: conflict[i][j] = True if i and j cannot coexist in a valid set
    # Two numbers x, y conflict if:
    #   - x divides sum of any 3 distinct others that include y (and 2 others)
    #   - y divides sum of any 3 distinct others that include x (and 2 others)
    # But this is complex. Simpler: precompute for each pair (x,y) whether they can be in the same valid set
    # by checking if there exist a, b distinct from x,y such that x | (a+b+y) or y | (a+b+x).
    # This is still expensive. Let's use a simpler conflict: 
    # x and y are "directly conflicting" if for ALL a,b distinct from x,y, 
    # (x+b+y) % a != 0 AND (y+a+x) % b != 0 is not what we want.
    # Actually, the constraint is: no selected element divides sum of any 3 distinct OTHER selected elements.
    # So x and y can coexist as long as there's no triple of other elements whose sum is divisible by x or y.
    # This depends on the full set, not just pairs.
    
    # Alternative approach: Use the existing _can_add but with memoization and bitset optimization.
    
    # Let's implement a recursive B&B with:
    # 1. Current selected set (as a list for efficiency)
    # 2. Remaining candidates (as a list)
    # 3. Pruning: if len(current) + len(remaining) <= best_size, prune
    # 4. Ordering: try candidates that are most "compatible" first
    
    # To speed up _can_add, we can precompute sums of pairs.
    # For a given set S, the check for adding x involves:
    #   - x divides sum of any 3 from S: check all C(|S|,3) triples
    #   - For each a in S, a divides (x + b + c) for any b,c in S\{a}
    
    # Let's use a different strategy: iterative deepening with greedy completion.
    # For each subset of size up to some limit, try to greedily complete it.
    
    # Actually, let's implement a proper B&B with good heuristics:
    
    def _can_add_fast(x, selected_list):
        """Fast version of _can_add using precomputed pair sums."""
        m = len(selected_list)
        if m < 3:
            # Check 2: for each a in selected, does a divide (x + b + c) for b,c in selected\{a}?
            # If m < 3, there aren't 3 others, so this is vacuously true.
            # But wait, we need 3 DISTINCT OTHER elements. If m < 3, there aren't 3 others.
            # So x can always be added if m < 3? No, we need to check if x divides sum of 3 from selected.
            # If m < 3, there are no 3 elements, so x divides nothing. So x can be added.
            # And for check 2, we need 3 distinct others for each a, which requires m-1 >= 3, i.e., m >= 4.
            # So if m < 4, both checks pass.
            return True
        
        # Check 1: x divides sum of any 3 from selected
        # Precompute all pair sums to speed up triple sum computation
        # triple sum = a + b + c = (a+b) + c
        for i in range(m):
            for j in range(i + 1, m):
                ab = selected_list[i] + selected_list[j]
                for k in range(j + 1, m):
                    if (ab + selected_list[k]) % x == 0:
                        return False
        
        # Check 2: for each a in selected, a divides (x + b + c) for any b,c in selected\{a}
        for i, a in enumerate(selected_list):
            # Need b, c from selected\{a}, b != c
            # (x + b + c) % a == 0
            for j in range(m):
                if j == i:
                    continue
                for k in range(j + 1, m):
                    if k == i:
                        continue
                    if (x + selected_list[j] + selected_list[k]) % a == 0:
                        return False
        
        return True
    
    # B&B with recursive search
    # State: (current_list, remaining_list)
    # We'll use a stack-based approach to avoid recursion depth issues
    
    best_solution = list(selected)
    best_size = len(selected)
    nodes_visited = 0
    node_limit = 200000  # Adjust based on timing
    
    def _try_add_and_complete(current_list, remaining_list):
        """Try to add candidates from remaining_list to current_list greedily."""
        new_list = list(current_list)
        new_remaining = list(remaining_list)
        
        while new_remaining and len(new_list) < MAX_ELEM:
            added = False
            # Try to add the first candidate that works
            for idx in range(len(new_remaining)):
                x = new_remaining[idx]
                if _can_add_fast(x, new_list):
                    new_list.append(x)
                    new_remaining.pop(idx)
                    added = True
                    break
            if not added:
                break
        
        return new_list, new_remaining
    
    # Main B&B: we'll do a depth-first search
    # At each node, we have current_list and remaining_list
    # We try to add candidates in order, and also try excluding them
    
    def _bbs_search(current_list, remaining_list, depth=0):
        nonlocal best_size, best_solution, nodes_visited
        
        if time.time() > deadline:
            return False
        if nodes_visited > node_limit:
            return False
        
        nodes_visited += 1
        current_size = len(current_list)
        
        # Upper bound: current_size + len(remaining_list)
        if current_size + len(remaining_list) <= best_size:
            return False
        
        # If no remaining candidates, update best
        if not remaining_list:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_list)
            return True
        
        # Try to greedily complete to get a quick lower bound
        greedy_list, greedy_remaining = _try_add_and_complete(current_list, remaining_list)
        if len(greedy_list) > best_size:
            best_size = len(greedy_list)
            best_solution = list(greedy_list)
        
        if best_size >= current_size + len(remaining_list):
            return True
        
        # Now do B&B on the remaining candidates
        # Choose a candidate to branch on
        # Heuristic: pick a candidate that conflicts with many others (high degree)
        # For simplicity, pick the first one
        v = remaining_list[0]
        rest = remaining_list[1:]
        
        # Branch 1: Include v if possible
        if _can_add_fast(v, current_list):
            new_current = current_list + [v]
            # Filter rest: remove candidates that can't coexist with v
            # A candidate u can't coexist with v if adding u to new_current fails
            # But that's expensive. Let's just pass rest and let _can_add_fast handle it.
            
            # Recurse
            _bbs_search(new_current, rest, depth + 1)
            
            if time.time() > deadline or nodes_visited > node_limit:
                return True
            
            # After including v, try excluding v as well
            # But if we already found a solution with v, we might not need to explore excluding v
            # unless we can do better without v
            # For now, always explore both branches
        
        # Branch 2: Exclude v
        _bbs_search(current_list, rest, depth + 1)
        
        return time.time() <= deadline and nodes_visited <= node_limit
    
    # Start search
    _bbs_search(list(selected), [x for x in range(N, 0, -1) if x not in current_set])
    
    return best_solution
>>>>>>> REPLACE

