import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_rows = None
    best_max = float('inf')
    
    # 1. Initial Greedy Solution with Bitmask
    # This provides a baseline upper bound.
    # We try a few variations of greedy (though standard greedy is deterministic, 
    # we can simulate diversity by slightly randomizing the choice among valid 
    # small candidates in a "relaxed greedy" mode, but standard greedy is usually 
    # a strong starting point for the backtracker to improve upon).
    
    def get_greedy_solution(seed=0):
        rng = random.Random(seed)
        used_diffs = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check conflicts with used_diffs
                # Diffs are x - a for a in row
                conflict = False
                for a in row:
                    d = x - a
                    if (used_diffs >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    # Update used_diffs
                    for a in row[:-1]: # a is in row, x is new last
                        d = x - a
                        used_diffs |= (1 << d)
            rows.append(row)
        return rows

    # Try a few random restarts for greedy to see if we get a better initial bound
    # Standard greedy is deterministic, but randomizing the order of checking x might help?
    # Actually, standard greedy always picks the smallest valid x. 
    # A "randomized greedy" could pick from the first N valid candidates.
    # Let's stick to standard greedy first as it's very fast and usually good.
    greedy_rows = get_greedy_solution()
    best_max = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows

    # 2. Backtracking Solver with Bitmask
    # We search for a solution with max entry <= target_max.
    # We use bitmasks for the set of used differences.
    
    def try_construct(target_max, n_rows, k, random_state):
        used_diff_mask = 0
        rows = []
        
        # Precompute diff masks for all possible pairs (a, b) where 0 <= a < b <= target_max
        # This is O(target_max^2) which is ~134^2 = ~18k, very fast.
        # mask[a][b] = bitmask of diffs if we have row ... a, b ...? 
        # No, the diffs depend on all previous elements in the row.
        # However, we can precompute the bitmask of diffs for a single new element x 
        # given the current row prefix? No, that depends on the prefix.
        
        # Let's just compute on the fly but efficiently.
        # For a row prefix [a_0, ..., a_{i-1}] and candidate x, 
        # the new diffs are {x - a_j for j in 0..i-1}.
        # We can compute this mask quickly.
        
        def backtrack_row(idx, row, used_diff_mask):
            if idx == k:
                return row[:]
            
            last = row[-1]
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                # Check if adding x creates any conflict
                # Diffs are x - a for a in row
                # We need to check if any of these diffs are already in used_diff_mask
                # And they must be distinct (they are, since row is strictly increasing)
                
                conflict = False
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    if (used_diff_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs_mask |= (1 << d)
                
                if not conflict:
                    valid_candidates.append((x, new_diffs_mask))
                    
            if not valid_candidates:
                return None
            
            # Heuristic: Prefer smaller x, but randomize
            # We can shuffle a subset of the smallest candidates
            # Let's take the first W candidates (smallest x) and shuffle them
            W = 10
            candidates_to_try = valid_candidates[:W] if len(valid_candidates) > W else valid_candidates
            random_state.shuffle(candidates_to_try)
            
            for x, new_diffs_mask in candidates_to_try:
                new_row = row + [x]
                res = backtrack_row(idx + 1, new_row, used_diff_mask | new_diffs_mask)
                if res is not None:
                    return res
                    
            return None

        # Main loop for rows
        for i in range(n_rows):
            row = [0]
            res = backtrack_row(0, row, used_diff_mask)
            if res is None:
                return None
            # Update global mask
            # We need to calculate the full mask for the completed row
            # to add to used_diff_mask
            for j in range(len(res)):
                for l in range(j+1, len(res)):
                    d = res[l] - res[j]
                    used_diff_mask |= (1 << d)
            rows.append(res)
            
        return rows

    # 3. Local Improvement / Reconstruction
    # Try to rebuild individual rows to reduce the maximum entry.
    # This is useful when we have a solution with max M, and we want to see 
    # if we can reduce it to M-1 by rebuilding one row.
    
    def try_local_improve(rows, n_rows, k, random_state):
        current_max = max(max(r) for r in rows)
        improved = False
        
        # Try to reduce max by 1
        target = current_max - 1
        if target < k:
            return rows, current_max, False
            
        # We will try to rebuild each row in turn
        # We need the set of differences used by the OTHER rows
        # Note: The differences within the row being rebuilt must also be distinct 
        # and not conflict with the other rows.
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
                
            # Compute used diffs excluding this row
            used_diffs_mask = 0
            for i, r in enumerate(rows):
                if i == row_idx:
                    continue
                for j in range(len(r)):
                    for l in range(j+1, len(r)):
                        d = r[l] - r[j]
                        used_diffs_mask |= (1 << d)
            
            # Try to rebuild row_idx with max entry <= target
            # We use a similar backtracking approach but for a single row
            # constrained by used_diffs_mask
            
            def backtrack_single_row(idx, row, used_mask):
                if idx == k:
                    return row[:]
                
                last = row[-1]
                valid_candidates = []
                for x in range(last + 1, target + 1):
                    conflict = False
                    new_diffs_mask = 0
                    for a in row:
                        d = x - a
                        if (used_mask >> d) & 1:
                            conflict = True
                            break
                        new_diffs_mask |= (1 << d)
                    
                    if not conflict:
                        valid_candidates.append((x, new_diffs_mask))
                        
                if not valid_candidates:
                    return None
                
                # Heuristic: prefer smaller x
                W = 10
                candidates_to_try = valid_candidates[:W] if len(valid_candidates) > W else valid_candidates
                random_state.shuffle(candidates_to_try)
                
                for x, new_diffs_mask in candidates_to_try:
                    new_row = row + [x]
                    res = backtrack_single_row(idx + 1, new_row, used_mask | new_diffs_mask)
                    if res is not None:
                        return res
                return None

            new_row = backtrack_single_row(0, [0], used_diffs_mask)
            if new_row is not None:
                # Check if the new row actually reduces the max
                # It might not if other rows have max == current_max
                # But if we successfully built a row with max <= target, 
                # and the other rows have max <= current_max, 
                # the new global max is max(target, max(other rows)).
                # If target < current_max, and no other row has max > target, 
                # then we improved.
                
                # Actually, we just need to replace the row and check the new global max
                old_row = rows[row_idx]
                rows[row_idx] = new_row
                new_global_max = max(max(r) for r in rows)
                
                if new_global_max < current_max:
                    current_max = new_global_max
                    improved = True
                else:
                    # Revert if it didn't help
                    rows[row_idx] = old_row
                    
            if improved:
                # If we improved, we can try again to improve further?
                # Or just break and let the main loop handle further reductions?
                # Let's break to allow the main loop to update best_max and restart
                break
                
        return rows, current_max, improved

    # Main Search Loop
    # We want to minimize the maximum entry.
    # Start from best_max - 1 and go down.
    # If we find a solution, update best_max and continue.
    
    current_target = best_max - 1
    
    # We will use a randomized restart strategy for the backtracking solver.
    # If we fail to find a solution for a target, we try a few restarts.
    # If we still fail, we assume it's impossible (or too hard) and stop.
    
    while time.time() < deadline and current_target >= k:
        found = False
        # Try multiple restarts for this target
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            random_state = random.Random()
            res = try_construct(current_target, n_rows, k, random_state)
            if res is not None:
                current_max = max(max(r) for r in res)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = res
                found = True
                break
        
        if found:
            # If we found a solution, try to improve it locally
            if time.time() < deadline:
                improved_rows, improved_max, was_improved = try_local_improve(
                    [r[:] for r in best_rows], n_rows, k, random.Random()
                )
                if was_improved:
                    best_max = improved_max
                    best_rows = improved_rows
            
            current_target = best_max - 1
        else:
            # If we can't find a solution for current_target, 
            # it might be impossible or just hard.
            # Stop searching for lower targets.
            break

    return best_rows
