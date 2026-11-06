"""Rewritten program for Sidon set extension using conflict graph + branch-and-bound."""
import sys
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a maximal Sidon set in [1, 284] containing {1, 2, 5, 11}.
    
    Approach:
    1. Start with the prescribed elements {1, 2, 5, 11}.
    2. Build a conflict graph where vertices are candidate integers (12..284)
       and edges represent violations of the Sidon property.
    3. Use a greedy branch-and-bound heuristic to maximize the set size.
    
    The Sidon property requires that all pairwise sums a + b (a <= b) are distinct.
    For a candidate x to be added to set S, we need:
    - For all y in S: x + y is not already in the sum set
    - 2*x is not already in the sum set
    - Additionally, x must not conflict with any other candidate that might be added later
    """
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Compute initial sums from base elements
    initial_sums = set()
    for i in range(len(base)):
        for j in range(i, len(base)):
            initial_sums.add(base[i] + base[j])
    
    # Candidate elements: 12 to 284
    candidates = list(range(12, 285))
    n_cand = len(candidates)
    
    # For each candidate, determine which base elements it conflicts with
    # A candidate x is valid if all x+y (y in base) and 2*x are not in initial_sums
    valid_candidates = []
    for x in candidates:
        new_sums = set()
        for y in base:
            s = x + y
            if s in initial_sums:
                break
            new_sums.add(s)
        else:
            if 2 * x in initial_sums:
                break
            # Check all sums are new
            if not new_sums & initial_sums:
                valid_candidates.append(x)
    
    # Build conflict graph among valid_candidates
    # Two candidates x and y conflict if adding both would violate Sidon property
    # This happens if any of: x+y, 2*x, 2*y, x+z, y+z (z in base) create duplicate sums
    # More precisely, we need to check if the sum sets overlap
    
    m = len(valid_candidates)
    cand_list = valid_candidates
    
    # Precompute sums with base for each candidate
    cand_base_sums = []
    for x in cand_list:
        s = set()
        for y in base:
            s.add(x + y)
        s.add(2 * x)
        cand_base_sums.append(s)
    
    # Build conflict matrix
    # conflict[i][j] = True if cand_list[i] and cand_list[j] cannot both be in the set
    conflict = [[False] * m for _ in range(m)]
    
    for i in range(m):
        xi = cand_list[i]
        si = cand_base_sums[i]
        for j in range(i + 1, m):
            xj = cand_list[j]
            sj = cand_base_sums[j]
            
            # Check if any sum in si equals any sum in sj
            # Also need to check if xi + xj creates a conflict
            # The full sum set for {base, xi, xj} includes:
            # - initial_sums
            # - si (sums involving xi and base or xi itself)
            # - sj (sums involving xj and base or xj itself)
            # - xi + xj
            
            # Conflict if si and sj share any element, or if xi+xj is in si or sj or initial_sums
            if si & sj:
                conflict[i][j] = True
                conflict[j][i] = True
            else:
                cross_sum = xi + xj
                if cross_sum in initial_sums or cross_sum in si or cross_sum in sj:
                    conflict[i][j] = True
                    conflict[j][i] = True
    
    # Now solve Maximum Independent Set on this conflict graph
    # Using a greedy branch-and-bound approach
    
    # Convert to adjacency list
    adj = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if conflict[i][j]:
                adj[i].append(j)
                adj[j].append(i)
    
    # Degree-based ordering for better pruning
    # Use a simple greedy MIS algorithm with backtracking
    
    best_size = 0
    best_set = set()
    
    # Greedy initial solution
    greedy_set = set()
    remaining = set(range(m))
    
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        min_deg = float('inf')
        chosen = -1
        for v in remaining:
            deg = sum(1 for u in adj[v] if u in remaining)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        if chosen == -1:
            break
        greedy_set.add(chosen)
        remaining.discard(chosen)
        # Remove neighbors of chosen
        for u in adj[chosen]:
            if u in remaining:
                remaining.discard(u)
    
    best_size = len(greedy_set)
    best_set = greedy_set.copy()
    
    # Branch and bound to improve
    # We'll use a recursive approach with upper bounds
    
    # For upper bound: use the fact that in a graph with max degree d,
    # MIS size >= n / (d+1), but we want an upper bound for pruning
    # Simple upper bound: number of remaining vertices
    
    def branch_and_bound(available, current_set, current_size):
        nonlocal best_size, best_set
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            return
        
        # Upper bound: current_size + len(available)
        if current_size + len(available) <= best_size:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with maximum degree in available subgraph for better pruning
        max_deg = -1
        chosen = -1
        for v in available:
            deg = sum(1 for u in adj[v] if u in available)
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        if chosen == -1:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            return
        
        # Branch 1: Include chosen
        new_available = set(available)
        new_available.discard(chosen)
        for u in adj[chosen]:
            new_available.discard(u)
        
        current_set.add(chosen)
        branch_and_bound(new_available, current_set, current_size + 1)
        current_set.discard(chosen)
        
        # Branch 2: Exclude chosen
        new_available = set(available)
        new_available.discard(chosen)
        branch_and_bound(new_available, current_set, current_size)
    
    # Start branch and bound from the greedy solution's complement
    # Actually, let's just run B&B from scratch with a timeout-aware approach
    
    start_time = time.time()
    timeout = 110  # Leave some margin for the 120s limit
    
    # Use a simpler approach: iterated local search
    # Start with greedy, then try to improve by swapping
    
    current_set = greedy_set.copy()
    current_size = len(current_set)
    
    # Try to add vertices that don't conflict with current set
    for v in range(m):
        if v in current_set:
            continue
        # Check if v conflicts with any vertex in current_set
        conflicts = False
        for u in current_set:
            if conflict[v][u]:
                conflicts = True
                break
        if not conflicts:
            current_set.add(v)
            current_size += 1
    
    if current_size > best_size:
        best_size = current_size
        best_set = current_set.copy()
    
    # Local search: try to swap vertices to increase size
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        for v in list(current_set):
            for w in range(m):
                if w in current_set:
                    continue
                # Try removing v and adding w
                new_set = current_set - {v}
                # Check if w can be added
                can_add = True
                for u in new_set:
                    if conflict[w][u]:
                        can_add = False
                        break
                if can_add:
                    # This swap keeps size same, but might enable future additions
                    # Let's just try adding w directly first
                    pass
        
        # Try to add any non-conflicting vertex
        for w in range(m):
            if w in current_set:
                continue
            can_add = True
            for u in current_set:
                if conflict[w][u]:
                    can_add = False
                    break
            if can_add:
                current_set.add(w)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
    
    # Convert best_set (indices) to actual values
    result = list(base)
    for idx in best_set:
        result.append(cand_list[idx])
    
    # Sort for consistency
    result.sort()
    
    return result

# EVOLVE-BLOCK-END
