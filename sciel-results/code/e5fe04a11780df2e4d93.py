import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total distinct differences needed
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    lower_bound_M = total_diffs_needed
    
    # Heuristic upper bound. 
    # For (4,7), LB=112, Best known=134.
    # We search M from LB upwards.
    # We will use a CDVR (Conflict-Directed Value Removal) local search.
    
    max_M_to_try = lower_bound_M + 60  # Search a bit beyond best known if needed
    
    # Helper to check if a set of rows is valid and return the scope
    def check_validity(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False, 0
                    used.add(d)
        return True, max(max(r) for r in rows)

    # Helper to initialize rows greedily with randomization
    def greedy_init(M):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Randomize the starting point of the search for the next value
                # to escape local minima of the simple greedy
                # We try values from current_max+1 to M
                # To introduce diversity, we can shuffle the candidate range
                # But we must maintain increasing order.
                # Simple approach: try values in random order within [current_max+1, M - (k+1-len(row))]
                
                min_val = current_max + 1
                max_val = M - (k + 1 - len(row))
                
                if min_val > max_val:
                    return None # Failed to fit in M
                
                candidates = list(range(min_val, max_val + 1))
                random.shuffle(candidates)
                
                found = False
                for val in candidates:
                    valid = True
                    new_diffs = []
                    for prev_val in row:
                        d = val - prev_val
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(val)
                        used_diffs.update(new_diffs)
                        current_max = val
                        found = True
                        break
                
                if not found:
                    return None
            
            rows.append(row)
        
        return rows

    # CDVR Local Search
    def cdvr_local_search(rows, M, time_budget):
        """
        rows: list of lists, each of length k+1, starting with 0.
        M: max allowed value.
        time_budget: seconds.
        Returns: (rows, scope) if valid, else (None, None)
        """
        start_ls = time.time()
        
        # Precompute candidates for each position? 
        # Since M is small, we can just check on the fly.
        
        # We maintain the set of used differences and a count of conflicts for each diff
        # Actually, to be efficient, we track:
        # 1. used_diffs: set of all differences currently in the rows
        # 2. diff_counts: dict mapping diff -> count of occurrences (should be 1 if valid, >1 if conflict)
        #    Actually, for CDVR, we need to know which specific row/pos pair is causing a conflict.
        
        # Let's maintain a list of all differences and their source (row_idx, i, j)
        # This might be too heavy. 
        # Alternative: 
        # 1. Compute all differences.
        # 2. Identify conflicts (differences appearing > 1 time).
        # 3. Pick a conflict difference d.
        # 4. Pick one of the rows/positions involved in d.
        # 5. Try to change that position to a new value v' such that:
        #    - v' is valid (increasing order)
        #    - v' <= M
        #    - The new differences introduced by v' do not create NEW conflicts (or reduce total conflicts).
        #    - Ideally, the new differences are not in used_diffs.
        
        # For speed, we will use a simpler heuristic:
        # 1. Find all conflicts.
        # 2. If no conflicts, return success.
        # 3. Pick a random conflict difference d.
        # 4. Pick a random row r and position p involved in d.
        # 5. Try to swap row[r][p] with a new value from a candidate pool.
        #    Candidate pool: values that keep the row increasing and <= M.
        #    We prioritize values that don't create new conflicts.
        
        rows = [r[:] for r in rows] # Deep copy
        
        # Helper to recompute conflicts
        def get_conflicts():
            diff_sources = {}
            for r_idx, row in enumerate(rows):
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d not in diff_sources:
                            diff_sources[d] = []
                        diff_sources[d].append((r_idx, i, j))
            
            conflicts = []
            for d, sources in diff_sources.items():
                if len(sources) > 1:
                    conflicts.append((d, sources))
            return conflicts

        # Initial check
        conflicts = get_conflicts()
        if not conflicts:
            return rows, max(max(r) for r in rows)
        
        while time.time() < start_ls + time_budget:
            # Pick a conflict
            d, sources = random.choice(conflicts)
            
            # Pick a source to modify
            # We want to modify a value. The value is either row[i] or row[j].
            # Modifying row[j] (the larger one) is usually easier to keep increasing order?
            # Actually, modifying any element requires checking neighbors.
            # Let's pick a random source (r_idx, i, j).
            r_idx, i, j = random.choice(sources)
            
            # We will try to change rows[r_idx][j] (the upper bound of the diff)
            # Why j? Because it has fewer constraints on the upper side (only M) 
            # compared to i which has constraints on both sides?
            # Actually, changing an intermediate element is complex.
            # Let's try changing the last element first? No, random is better.
            # Let's try changing rows[r_idx][j].
            
            old_val = rows[r_idx][j]
            
            # Determine valid range for new value at index j in row r_idx
            # Must be > rows[r_idx][j-1] and < rows[r_idx][j+1] (if exists)
            # And <= M
            min_new = rows[r_idx][j-1] + 1
            max_new = M
            if j < len(rows[r_idx]) - 1:
                max_new = min(max_new, rows[r_idx][j+1] - 1)
            
            if min_new > max_new:
                # Can't change this position. Try another source?
                # For simplicity, skip this iteration
                continue
            
            # Generate candidates
            # To speed up, we don't try ALL values. We try a random subset.
            # But we must ensure we don't create new conflicts.
            
            # A good heuristic: Try values that are "far" from other elements to avoid collisions?
            # Or just random.
            
            candidates = list(range(min_new, max_new + 1))
            random.shuffle(candidates)
            
            # Limit candidates to avoid O(M^2) in worst case per step
            # M is ~130, so 130 candidates is fine.
            
            moved = False
            for new_val in candidates:
                if new_val == old_val:
                    continue
                
                # Check if new_val creates increasing order violation
                # (Already handled by min_new/max_new)
                
                # Check if new_val creates any NEW conflicts
                # We need to calculate the new differences involving index j.
                # Old diffs involving j: (old_val - rows[r_idx][i']) for i' < j
                # New diffs involving j: (new_val - rows[r_idx][i']) for i' < j
                # Also, if j is not the last, diffs with i' > j are (rows[r_idx][i'] - old_val)
                # Wait, the diff is defined as row[j]-row[i] for i<j.
                # So if we change row[j], it affects diffs (j, i) for all i < j.
                # It does NOT affect diffs (i, j) for i > j because those are (row[i]-row[j])?
                # No, the definition is "differences between two entries ... a_j - a_i (i < j)".
                # So if we change element at index `pos`, it affects:
                # 1. Diffs where this element is the `j` (larger index): d = val - row[i] for i < pos
                # 2. Diffs where this element is the `i` (smaller index): d = row[j] - val for j > pos
                
                # Let's verify this logic.
                # Yes.
                
                # To check validity efficiently:
                # 1. Remove old diffs involving `pos` from a temporary count map?
                #    This is expensive if we do it from scratch.
                #    Instead, we can check if the NEW diffs are already present in the global used set
                #    (excluding the ones we are about to remove).
                
                # Let's build the set of diffs we are removing.
                remove_diffs = set()
                for i in range(len(rows[r_idx])):
                    if i == j: continue
                    if i < j:
                        remove_diffs.add(old_val - rows[r_idx][i])
                    else:
                        remove_diffs.add(rows[r_idx][i] - old_val)
                
                # Build the set of new diffs we are adding.
                add_diffs = set()
                for i in range(len(rows[r_idx])):
                    if i == j: continue
                    if i < j:
                        add_diffs.add(new_val - rows[r_idx][i])
                    else:
                        add_diffs.add(rows[r_idx][i] - new_val)
                
                # A move is "safe" if none of the add_diffs are in the global used set
                # MINUS the remove_diffs.
                # i.e., add_diffs ∩ (used_diffs \ remove_diffs) == empty
                
                # We need the global used_diffs. We can maintain it incrementally.
                # For now, let's just recompute or maintain a global set.
                # Maintaining a global set is better.
                
                # Let's assume we have a global `current_used_diffs` set.
                # We check if any d in add_diffs is in current_used_diffs and NOT in remove_diffs.
                
                is_safe = True
                for d in add_diffs:
                    if d in current_used_diffs and d not in remove_diffs:
                        is_safe = False
                        break
                
                if is_safe:
                    # Apply the move
                    # Update current_used_diffs
                    current_used_diffs.difference_update(remove_diffs)
                    current_used_diffs.update(add_diffs)
                    
                    # Update the row
                    rows[r_idx][j] = new_val
                    
                    moved = True
                    break
            
            if not moved:
                # No safe move found for this conflict. 
                # We could try a different conflict or accept a "worse" move (not implemented here).
                # To avoid infinite loops on unsolvable states, we might need to backtrack or restart.
                # For now, just pick a new conflict.
                pass
            
            # Update conflicts for next iteration
            # Instead of full recompute, we could update incrementally, but full recompute is O(n*k^2) which is small.
            conflicts = get_conflicts()
            if not conflicts:
                return rows, max(max(r) for r in rows)
                
        return None, None

    # Main loop
    best_rows = None
    best_scope = float('inf')
    
    for M in range(lower_bound_M, max_M_to_try + 1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations for this M
        found = False
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
                
            rows = greedy_init(M)
            if rows is None:
                continue
            
            # Check if already valid
            valid, scope = check_validity(rows)
            if valid:
                return rows
            
            # Run CDVR
            # We need to initialize current_used_diffs for CDVR
            # The CDVR function above assumes `current_used_diffs` exists in closure or is passed.
            # Let's refactor CDVR to take the initial rows and compute the set internally.
            
            # Refactored CDVR for clarity and correctness
            def cdvr_refactored(rows, M, time_budget):
                start_ls = time.time()
                rows = [r[:] for r in rows]
                
                # Build initial used_diffs
                current_used_diffs = set()
                for row in rows:
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            current_used_diffs.add(row[j] - row[i])
                
                def get_conflicts():
                    diff_sources = {}
                    for r_idx, row in enumerate(rows):
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                d = row[j] - row[i]
                                if d not in diff_sources:
                                    diff_sources[d] = []
                                diff_sources[d].append((r_idx, i, j))
                    
                    conflicts = []
                    for d, sources in diff_sources.items():
                        if len(sources) > 1:
                            conflicts.append((d, sources))
                    return conflicts

                conflicts = get_conflicts()
                if not conflicts:
                    return rows, max(max(r) for r in rows)
                
                while time.time() < start_ls + time_budget:
                    d, sources = random.choice(conflicts)
                    r_idx, i, j = random.choice(sources)
                    
                    # Try to change rows[r_idx][j]
                    old_val = rows[r_idx][j]
                    
                    min_new = rows[r_idx][j-1] + 1
                    max_new = M
                    if j < len(rows[r_idx]) - 1:
                        max_new = min(max_new, rows[r_idx][j+1] - 1)
                    
                    if min_new > max_new:
                        continue
                    
                    candidates = list(range(min_new, max_new + 1))
                    random.shuffle(candidates)
                    
                    moved = False
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        
                        remove_diffs = set()
                        for idx in range(len(rows[r_idx])):
                            if idx == j: continue
                            if idx < j:
                                remove_diffs.add(old_val - rows[r_idx][idx])
                            else:
                                remove_diffs.add(rows[r_idx][idx] - old_val)
                        
                        add_diffs = set()
                        for idx in range(len(rows[r_idx])):
                            if idx == j: continue
                            if idx < j:
                                add_diffs.add(new_val - rows[r_idx][idx])
                            else:
                                add_diffs.add(rows[r_idx][idx] - new_val)
                        
                        is_safe = True
                        for d_new in add_diffs:
                            if d_new in current_used_diffs and d_new not in remove_diffs:
                                is_safe = False
                                break
                        
                        if is_safe:
                            current_used_diffs.difference_update(remove_diffs)
                            current_used_diffs.update(add_diffs)
                            rows[r_idx][j] = new_val
                            moved = True
                            break
                    
                    if not moved:
                        continue
                        
                    conflicts = get_conflicts()
                    if not conflicts:
                        return rows, max(max(r) for r in rows)
                        
                return None, None

            res_rows, res_scope = cdvr_refactored(rows, M, 5.0) # 5s budget per attempt
            if res_rows is not None:
                return res_rows

    # Fallback: Greedy
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    used_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    break
                x += 1
        rows.append(row)
        
    return rows
