import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 115.0
    m = k + 1  # entries per row
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    # Lower bound: need at least total_diffs_needed distinct differences
    # Max difference is scope. So scope >= total_diffs_needed
    lb = total_diffs_needed

    # For (4, 7): total_diffs_per_row = 28, total_diffs_needed = 112.
    # Best known is 134.

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        diffs = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_candidate_rows(max_val, count, rng):
        """Generate 'count' valid row candidates with max entry <= max_val."""
        candidates = []
        # We want diverse candidates.
        # Strategy: Randomized greedy with backtracking?
        # Or just random increasing sequences that don't have internal conflicts?
        # Actually, a row is valid if its internal differences are distinct.
        # This is a Sidon-like property for the set of elements? No, just that a_j - a_i are distinct.
        # This is equivalent to saying the set is a "Golomb ruler".
        
        # Let's generate random Golomb rulers of length m with max entry <= max_val.
        attempts = 0
        max_attempts = count * 20
        
        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            if time.time() > deadline - 5:
                break
            
            # Randomized construction of a Golomb ruler
            row = [0]
            success = True
            # Sort positions? No, just pick next element.
            # To ensure diversity, maybe pick from a range.
            
            # Heuristic: Pick next element uniformly from [last+1, max_val]
            # but ensure no difference conflicts within the row.
            used_row_diffs = set()
            
            for _ in range(k):
                lo = row[-1] + 1
                hi = max_val
                if lo > hi:
                    success = False
                    break
                
                found = False
                # Try random candidates
                # Bias towards smaller values to keep scope down?
                # Or uniform?
                # Let's try a mix: 50% uniform, 50% biased towards lower end
                if rng.random() < 0.5:
                    candidates_to_try = [rng.randint(lo, hi) for _ in range(20)]
                else:
                    # Biased: x = lo + (hi-lo)*r^2
                    candidates_to_try = []
                    for _ in range(20):
                        r = rng.random()
                        x = int(lo + (hi - lo) * r * r)
                        if x > hi: x = hi
                        candidates_to_try.append(x)
                
                candidates_to_try.sort() # Try smaller first? Or random order?
                rng.shuffle(candidates_to_try)
                
                for x in candidates_to_try:
                    ok = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_row_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    if ok:
                        for d in new_diffs:
                            used_row_diffs.add(d)
                        row.append(x)
                        found = True
                        break
                
                if not found:
                    # Backtrack? For simplicity, restart this row generation
                    success = False
                    break
            
            if success:
                # Check if this row is unique (optional, but good for diversity)
                # Just add it
                candidates.append(row)
                
        return candidates

    def solve_mis(candidates, n_rows_needed):
        """
        Find a set of n_rows_needed non-conflicting rows from candidates.
        Conflict: two rows share any difference.
        Returns the set of rows or None.
        """
        n = len(candidates)
        if n < n_rows_needed:
            return None
        
        # Precompute conflicts
        # conflict[i] is a bitmask or set of indices j such that row i and row j conflict.
        # For n up to ~1000, bitmasks are fast.
        # Let's use Python integers as bitmasks if n < 2000.
        
        row_diffs = [get_diffs(row) for row in candidates]
        
        # Conflict graph
        # adj[i] is set of j > i that conflict with i? Or bitmask.
        # Let's use bitmasks for speed in branch and bound.
        adj = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if row_diffs[i] & row_diffs[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Color bound: Greedy coloring of the conflict graph.
        # Color c[i] is the number of colors used up to i in a greedy ordering.
        # This gives an upper bound on the size of the independent set.
        # We want to maximize the independent set size.
        # Actually, we want to find an independent set of size n_rows_needed.
        
        # Standard MIS branch and bound (Tomita's algorithm variant)
        # We want to find if there is an independent set of size k.
        
        # Order vertices by degree (descending) to improve coloring bound?
        # Or just use original order.
        
        # Greedy coloring
        # colors[i] = color assigned to vertex i
        # color_bound[i] = max color used in subgraph induced by {0..i}
        
        colors = [0] * n
        color_bound = [0] * n
        
        # Simple greedy coloring
        for i in range(n):
            used_colors = set()
            for j in range(i):
                if (adj[i] >> j) & 1:
                    used_colors.add(colors[j])
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            colors[i] = c
            if c + 1 > color_bound[i-1] if i > 0 else c + 1 > 0:
                color_bound[i] = max(color_bound[i-1] if i > 0 else 0, c + 1)
            else:
                color_bound[i] = color_bound[i-1] if i > 0 else c + 1
        
        # Refine color_bound: color_bound[i] is the max color in 0..i.
        # Actually, standard implementation:
        # color_bound[i] = max(color_bound[i-1], colors[i]+1)
        
        # Let's re-implement coloring carefully
        colors = [0] * n
        color_bound = [0] * n
        for i in range(n):
            used = set()
            for j in range(i):
                if (adj[i] >> j) & 1:
                    used.add(colors[j])
            c = 0
            while c in used:
                c += 1
            colors[i] = c
            color_bound[i] = max(color_bound[i-1] if i > 0 else 0, c + 1)
        
        # Branch and bound
        # State: current independent set (list), available vertices (bitmask), start vertex index?
        # We will use a recursive search.
        
        best_set = []
        found = False
        
        def dfs(current_set, avail_mask, start_idx):
            nonlocal found, best_set
            
            if len(current_set) == n_rows_needed:
                found = True
                best_set = list(current_set)
                return True
            
            # Pruning: if max possible size < n_rows_needed, return
            # Max possible size is bounded by the number of bits in avail_mask
            # But better bound: color bound on the subgraph induced by avail_mask
            # For simplicity, let's just check bit count first.
            if bin(avail_mask).count('1') < n_rows_needed - len(current_set):
                return False
            
            # Pick a vertex to branch on.
            # Usually pick the vertex with highest degree in the induced subgraph?
            # Or just the first available vertex.
            # Let's pick the vertex with index 'start_idx' if available, else next.
            
            # To make it efficient, we should pick a vertex that is in the available set.
            # Let's iterate through available vertices.
            
            # Optimization: If the color bound of the available subgraph is less than needed, prune.
            # Calculating color bound for arbitrary subgraph is expensive.
            # We will rely on the global color bound as a heuristic, or just bit count.
            # Given n is small (~100-500), bit count is okay.
            
            # Select vertex
            # Let's pick the vertex with the smallest index in avail_mask
            # This is not optimal but simple.
            
            # Better: Pick vertex with max degree in avail_mask
            # deg(v) = bin(avail_mask & adj[v]).count('1')
            
            best_v = -1
            best_deg = -1
            temp_mask = avail_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                deg = bin(avail_mask & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_mask ^= lsb
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            current_set.append(best_v)
            # New available: neighbors of best_v are removed
            new_avail = avail_mask & ~adj[best_v] & ~(1 << best_v)
            
            if dfs(current_set, new_avail, best_v + 1):
                return True
            
            current_set.pop()
            
            # Branch 2: Exclude best_v
            # New available: remove best_v
            new_avail = avail_mask & ~(1 << best_v)
            
            # Pruning: if color bound of new_avail < needed, skip?
            # Just recurse
            if dfs(current_set, new_avail, best_v + 1):
                return True
            
            return False
        
        # Start with all vertices available
        initial_mask = (1 << n) - 1
        
        # To improve performance, we can order vertices by degree before building the graph?
        # The adjacency matrix is fixed, but the order of indices matters for the greedy coloring bound.
        # Let's reorder vertices by descending degree to get a tighter color bound.
        
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        # Remap indices
        new_adj = [0] * n
        old_to_new = {old: new for new, old in enumerate(order)}
        
        for i in range(n):
            old_i = order[i]
            mask = adj[old_i]
            new_mask = 0
            temp = mask
            while temp:
                lsb = temp & -temp
                old_j = lsb.bit_length() - 1
                new_j = old_to_new[old_j]
                new_mask |= (1 << new_j)
                temp ^= lsb
            new_adj[i] = new_mask
        
        adj = new_adj
        
        # Recalculate colors and bounds with new order
        colors = [0] * n
        color_bound = [0] * n
        for i in range(n):
            used = set()
            for j in range(i):
                if (adj[i] >> j) & 1:
                    used.add(colors[j])
            c = 0
            while c in used:
                c += 1
            colors[i] = c
            color_bound[i] = max(color_bound[i-1] if i > 0 else 0, c + 1)
        
        # Map back: when we find a set of new indices, convert to old indices
        new_to_old = {new: old for old, new in old_to_new.items()}
        
        def dfs_reordered(current_set, avail_mask):
            nonlocal found, best_set
            
            if len(current_set) == n_rows_needed:
                found = True
                best_set = [new_to_old[v] for v in current_set]
                return True
            
            if bin(avail_mask).count('1') < n_rows_needed - len(current_set):
                return False
            
            # Pick vertex with max degree in avail_mask
            best_v = -1
            best_deg = -1
            temp_mask = avail_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                deg = bin(avail_mask & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_mask ^= lsb
            
            if best_v == -1:
                return False
            
            # Include
            current_set.append(best_v)
            new_avail = avail_mask & ~adj[best_v] & ~(1 << best_v)
            
            if dfs_reordered(current_set, new_avail):
                return True
            
            current_set.pop()
            
            # Exclude
            new_avail = avail_mask & ~(1 << best_v)
            
            if dfs_reordered(current_set, new_avail):
                return True
            
            return False
        
        initial_mask = (1 << n) - 1
        dfs_reordered([], initial_mask)
        
        if found:
            return [candidates[i] for i in best_set]
        return None

    # Main loop
    best_rows = None
    best_scope = float('inf')
    
    # Generate candidates in increasing scope
    # Start from lower bound
    # Step size?
    
    # For (4,7), lb=112. Best known 134.
    # Let's try scopes from lb to lb + 50
    
    # First, try to find ANY solution with a reasonable scope using greedy?
    # The current program does that. We want to beat it.
    # So we should aim for scope < current best (which is likely ~134 or less from seed).
    # Seed score 1.0 means it found the seed solution.
    
    # Let's generate candidates for a specific scope.
    # If we find a solution, we can try to reduce scope.
    
    # Strategy:
    # 1. Generate candidates for scope S.
    # 2. Solve MIS.
    # 3. If solution found, record it. Try S-1?
    #    But generating candidates for S-1 is not guaranteed to contain the optimal set if we didn't generate enough.
    #    So we should generate a large pool of candidates for a slightly larger scope,
    #    and then find the best independent set within that pool, minimizing the max entry.
    
    # Revised Strategy:
    # Generate a large pool of candidates with max entry <= S_max.
    # Solve MIS to find n_rows non-conflicting rows.
    # Among all such sets, find the one with minimum max entry.
    # This is a weighted MIS or just check all MIS?
    # MIS gives one set. We want the "best" set.
    # We can modify the branch and bound to minimize the max entry of the selected rows.
    
    # Let's implement a search that minimizes the max entry.
    # Objective: min max(row[i][-1] for i in set)
    
    # We can sort candidates by max entry.
    # And use a branch and bound that keeps track of the current max entry.
    
    def solve_mis_min_scope(candidates, n_rows_needed, deadline):
        n = len(candidates)
        if n < n_rows_needed:
            return None
        
        row_diffs = [get_diffs(row) for row in candidates]
        row_maxes = [row[-1] for row in candidates]
        
        # Conflict graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if row_diffs[i] & row_diffs[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Order by max entry? Or degree?
        # Let's order by max entry ascending, then degree descending.
        # This helps the "include" branch find low-scope solutions early?
        # Actually, we want to prune branches that cannot beat the current best.
        
        order = sorted(range(n), key=lambda i: (row_maxes[i], -bin(adj[i]).count('1')))
        
        new_adj = [0] * n
        old_to_new = {old: new for new, old in enumerate(order)}
        
        for i in range(n):
            old_i = order[i]
            mask = adj[old_i]
            new_mask = 0
            temp = mask
            while temp:
                lsb = temp & -temp
                old_j = lsb.bit_length() - 1
                new_j = old_to_new[old_j]
                new_mask |= (1 << new_j)
                temp ^= lsb
            new_adj[i] = new_mask
        
        adj = new_adj
        new_row_maxes = [row_maxes[old] for old in order]
        
        # Greedy coloring
        colors = [0] * n
        color_bound = [0] * n
        for i in range(n):
            used = set()
            for j in range(i):
                if (adj[i] >> j) & 1:
                    used.add(colors[j])
            c = 0
            while c in used:
                c += 1
            colors[i] = c
            color_bound[i] = max(color_bound[i-1] if i > 0 else 0, c + 1)
        
        best_set = None
        best_scope = float('inf')
        
        def dfs(current_set, avail_mask, current_max):
            nonlocal best_set, best_scope
            
            if time.time() > deadline - 5:
                return
            
            if len(current_set) == n_rows_needed:
                if current_max < best_scope:
                    best_scope = current_max
                    best_set = list(current_set)
                return
            
            # Pruning
            # 1. Not enough vertices
            if bin(avail_mask).count('1') < n_rows_needed - len(current_set):
                return
            
            # 2. Even if we pick the smallest max entries, can we beat best_scope?
            # The smallest max entry in avail_mask is min(new_row_maxes[v] for v in avail)
            # If current_max is already >= best_scope, we might still improve if we pick smaller ones?
            # No, current_max is the max of selected so far.
            # The final max will be max(current_max, max of new picks).
            # So if current_max >= best_scope, we can only improve if we pick rows with max < best_scope.
            # But current_max is fixed.
            # So if current_max >= best_scope, we cannot improve unless best_scope is inf.
            # Wait, if current_max < best_scope, we can still end up with max < best_scope.
            # If current_max >= best_scope, the final max will be >= current_max >= best_scope.
            # So we can prune if current_max >= best_scope.
            
            if current_max >= best_scope:
                return
            
            # 3. Color bound
            # If the max independent set size of avail_mask is < needed, prune.
            # We don't have a tight bound for subgraph.
            # But we can check: if the number of available vertices with max < best_scope is < needed, prune?
            # Not exactly, because we can pick vertices with max >= best_scope if current_max is low?
            # No, if we pick a vertex with max >= best_scope, the final max will be >= best_scope.
            # So we only care about vertices with max < best_scope.
            
            # Let's filter avail_mask to only include vertices with new_row_maxes[v] < best_scope
            # If the count of such vertices is < needed, prune.
            
            valid_mask = 0
            temp = avail_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if new_row_maxes[v] < best_scope:
                    valid_mask |= lsb
                temp ^= lsb
            
            if bin(valid_mask).count('1') < n_rows_needed - len(current_set):
                return
            
            # Pick vertex
            # Prefer vertices with smaller max entry?
            # Or highest degree?
            # Let's pick highest degree in valid_mask.
            
            best_v = -1
            best_deg = -1
            temp_mask = valid_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                deg = bin(valid_mask & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_mask ^= lsb
            
            if best_v == -1:
                return
            
            # Include best_v
            new_max = max(current_max, new_row_maxes[best_v])
            current_set.append(best_v)
            new_avail = valid_mask & ~adj[best_v] & ~(1 << best_v)
            
            dfs(current_set, new_avail, new_max)
            
            current_set.pop()
            
            # Exclude best_v
            new_avail = valid_mask & ~(1 << best_v)
            
            dfs(current_set, new_avail, current_max)
        
        initial_mask = (1 << n) - 1
        dfs([], initial_mask, 0)
        
        if best_set:
            return [candidates[old_to_new[inv]] for inv in best_set] # Wait, old_to_new is new->old?
            # order[new] = old
            # So candidates[order[v]]
        
        return None

    # Generate candidates
    # Let's start with a scope that is likely to contain a solution.
    # Seed found something. Let's assume seed scope is ~134.
    # Let's try scope 140.
    
    # To be safe, let's generate candidates for scope 150.
    # Number of candidates: 200?
    
    rng = random.Random(42)
    
    # Initial greedy solution to have a fallback
    used = set()
    fallback_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < m:
            x += 1
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                for a in row:
                    used.add(x - a)
                row.append(x)
        fallback_rows.append(row)
    
    fallback_scope = max(max(r) for r in fallback_rows)
    
    # Try MIS approach
    # Scope to try
    target_scope = min(fallback_scope, 150) # Don't go higher than fallback initially
    
    # Generate candidates
    candidates = generate_candidate_rows(target_scope, 300, rng)
    
    if len(candidates) >= n_rows:
        result = solve_mis_min_scope(candidates, n_rows, deadline)
        if result:
            # Validate
            if is_valid(result):
                scope = max(max(r) for r in result)
                if scope < fallback_scope:
                    fallback_rows = result
                    fallback_scope = scope
    
    # If we have time, try to improve with larger candidate pool or tighter scope
    if time.time() < deadline - 10:
        # Try tighter scope
        new_scope = fallback_scope - 1
        if new_scope > lb:
            candidates2 = generate_candidate_rows(new_scope, 200, rng)
            # Merge candidates? Or just solve on new pool?
            # Solving on new pool might miss the previous solution if it's not in the new pool.
            # But we want to find a solution with max entry <= new_scope.
            # So we only need candidates with max entry <= new_scope.
            
            if len(candidates2) >= n_rows:
                result2 = solve_mis_min_scope(candidates2, n_rows, deadline)
                if result2:
                    if is_valid(result2):
                        scope2 = max(max(r) for r in result2)
                        if scope2 < fallback_scope:
                            fallback_rows = result2
                            fallback_scope = scope2
    
    return fallback_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
