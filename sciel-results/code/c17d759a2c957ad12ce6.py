import random
import time
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def get_row_diffs(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                diffs |= (1 << d)
        return diffs

    def is_valid_row(row, global_used_except_self):
        # Check internal distinctness and external distinctness
        row_diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if (row_diffs >> d) & 1:
                    return False, 0 # Internal conflict
                row_diffs |= (1 << d)
        if row_diffs & global_used_except_self:
            return False, 0 # External conflict
        return True, row_diffs

    def greedy_row(max_val, global_used, existing_vals=None):
        """
        Greedily construct a row of length k+1 starting with 0, 
        with all values <= max_val, avoiding conflicts with global_used.
        existing_vals: if provided, these are fixed prefix values.
        """
        if existing_vals is None:
            row = [0]
        else:
            row = list(existing_vals)
            
        # Calculate diffs from existing row
        current_diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                current_diffs |= (1 << (row[j] - row[i]))
                
        # Check if existing row is already valid internally
        # (Assuming it is, as we build incrementally)
        
        while len(row) < k + 1:
            last = row[-1]
            found = False
            # Try smallest possible next value
            for v in range(last + 1, max_val + 1):
                # Check conflicts with current row
                conflict = False
                new_diffs = 0
                for u in row:
                    d = v - u
                    if (current_diffs >> d) & 1:
                        conflict = True
                        break
                    new_diffs |= (1 << d)
                
                if conflict:
                    continue
                
                # Check conflicts with global
                if new_diffs & global_used:
                    continue
                
                # Accept
                row.append(v)
                current_diffs |= new_diffs
                found = True
                break
            
            if not found:
                return None
        return row

    def random_row(max_val, global_used, attempts=50):
        """
        Try to find a valid row with max <= max_val using randomized greedy.
        """
        for _ in range(attempts):
            row = [0]
            current_diffs = 0
            success = True
            for _ in range(k):
                last = row[-1]
                candidates = list(range(last + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    conflict = False
                    new_diffs = 0
                    for u in row:
                        d = v - u
                        if (current_diffs >> d) & 1:
                            conflict = True
                            break
                        new_diffs |= (1 << d)
                    if conflict:
                        continue
                    if new_diffs & global_used:
                        continue
                    row.append(v)
                    current_diffs |= new_diffs
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                # Final check
                valid, _ = is_valid_row(row, global_used)
                if valid:
                    return row
        return None

    def backtrack_solve(rows, row_idx, max_val, global_used):
        """
        Recursively try to fill rows[row_idx:] such that all rows are valid and max entry <= max_val.
        Returns True if solution found, updating rows in place.
        """
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            return True
        
        # Calculate used diffs from rows before row_idx
        used_before = 0
        for i in range(row_idx):
            used_before |= get_row_diffs(rows[i])
        
        # Try to find a valid row for this index
        # We use a randomized approach to pick candidates
        # To make it efficient, we limit the number of candidates tried per recursion level
        # We can shuffle the range or just iterate.
        # Since we are backtracking, we need to explore options.
        
        # Heuristic: Try smaller values first? Or random?
        # Randomization helps escape local minima in the row construction.
        
        # Generate candidates for this row
        # We can try to construct the row greedily but with randomization in choice of next element
        # If greedy fails, backtrack.
        
        # For simplicity and speed, let's try to generate a few random valid rows for this index
        # and recurse.
        
        candidates = []
        # Try a few random constructions
        for _ in range(3): # 3 attempts per node
            row = random_row(max_val, used_before, attempts=1)
            if row is not None:
                candidates.append(row)
        
        # If greedy worked, include it
        g_row = greedy_row(max_val, used_before)
        if g_row is not None:
            candidates.append(g_row)
        
        # Shuffle candidates to add randomness
        random.shuffle(candidates)
        
        for c_row in candidates:
            rows[row_idx] = c_row
            if backtrack_solve(rows, row_idx + 1, max_val, global_used):
                return True
        
        return False

    # 1. Initial Solution: Greedy with large scope
    init_scope = 250
    rows = []
    global_used = 0
    success = False
    for _ in range(50):
        rows = [None] * n_rows
        # Use backtracking to find a valid set of rows within init_scope
        if backtrack_solve(rows, 0, init_scope, 0):
            success = True
            break
    
    if not success:
        # Fallback: Very large scope with pure greedy (should always work)
        rows = []
        global_used = 0
        for i in range(n_rows):
            row = greedy_row(500, global_used)
            if row is None:
                row = [0] + list(range(1, k+1))
            rows.append(row)
            global_used |= get_row_diffs(row)
            
    best_rows = [list(r) for r in rows]
    best_max = get_max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_max = best_max

    # 2. Optimization Loop: Binary Search for Minimum Max
    # We know best_max is valid. We want to find the smallest M such that a solution exists with max <= M.
    # We can binary search between lower_bound and best_max.
    
    # To speed up the search, we can also use local search on the found solution to reduce max further
    # but binary search with backtracking is more systematic.
    
    # Let's perform a binary search for the optimal scope.
    low = lower_bound
    high = best_max
    
    # We will try to find a solution for a given scope M using backtracking.
    # To avoid getting stuck in bad local minima during backtracking, we randomize the order of rows?
    # The rows are indistinguishable, so we can fix the order but randomize the construction.
    
    def find_solution_for_scope(M):
        """
        Try to find a valid set of rows with max entry <= M.
        Returns list of rows or None.
        """
        # Multiple attempts with different random seeds/orderings
        for attempt in range(10):
            rows = [None] * n_rows
            if backtrack_solve(rows, 0, M, 0):
                return rows
        return None

    # Binary search
    # We check from high to low? Or standard binary search?
    # Standard binary search: check mid. If valid, high = mid. Else low = mid + 1.
    
    # However, backtracking can be slow. We should limit the time per check.
    # We can integrate a time check into backtrack_solve.
    
    # Let's modify backtrack_solve to be time-aware.
    def backtrack_solve_timed(rows, row_idx, max_val, time_limit):
        if time.time() > time_limit:
            return False
        if row_idx == n_rows:
            return True
        
        used_before = 0
        for i in range(row_idx):
            used_before |= get_row_diffs(rows[i])
        
        candidates = []
        # Try a few random constructions
        for _ in range(3): # Fewer attempts per node to save time
            row = random_row(max_val, used_before, attempts=1)
            if row is not None:
                candidates.append(row)
        
        g_row = greedy_row(max_val, used_before)
        if g_row is not None:
            candidates.append(g_row)
            
        random.shuffle(candidates)
        
        for c_row in candidates:
            rows[row_idx] = c_row
            if backtrack_solve_timed(rows, row_idx + 1, max_val, time_limit):
                return True
        return False

    def find_solution_for_scope_timed(M, time_limit):
        for attempt in range(5):
            rows = [None] * n_rows
            if backtrack_solve_timed(rows, 0, M, time_limit):
                return rows
        return None

    # Run binary search
    # We have about 100 seconds left.
    # Let's do a coarse binary search first.
    
    search_start_time = time.time()
    remaining_time = deadline - search_start_time - 5.0 # 5s margin
    
    # If we have enough time, do binary search
    if remaining_time > 10:
        low = lower_bound
        high = best_max
        
        # We can speed up by starting from high and decreasing?
        # No, binary search is better.
        
        # To avoid repeated expensive searches, we can cache results? 
        # But the search is stochastic, so caching is hard.
        
        # Let's just iterate.
        # We will try to find the minimal M.
        
        # Optimization: If we find a solution for M, we can try to improve it using local search?
        # Or just keep the solution.
        
        # Let's try a few steps of binary search.
        while low < high and (time.time() - search_start_time) < remaining_time - 5:
            mid = (low + high) // 2
            # Check if mid is feasible
            # Allocate half the remaining time to this check
            check_time_limit = time.time() + (remaining_time - (time.time() - search_start_time)) * 0.5
            sol = find_solution_for_scope_timed(mid, check_time_limit)
            if sol is not None:
                # Mid is feasible
                best_rows = [list(r) for r in sol]
                best_max = get_max_entry(best_rows)
                high = mid
            else:
                # Mid is not feasible (or timed out)
                low = mid + 1
        
        # After binary search, try to polish the best solution found
        if best_rows:
            current_rows = best_rows
            current_max = best_max
    
    # 3. Local Search to Polish
    # Try to reduce the max entry of the current best solution further.
    
    def try_shrink_row(row_idx, target_max, global_used_except_self):
        for _ in range(20):
            row = random_row(target_max, global_used_except_self, attempts=1)
            if row is not None:
                return row
        return None

    current_rows = [list(r) for r in best_rows]
    current_max = get_max_entry(current_rows)
    
    # Local search loop
    no_improve_count = 0
    while time.time() < deadline - 2.0:
        row_idx = random.randint(0, n_rows - 1)
        
        global_used_except = 0
        for i in range(n_rows):
            if i != row_idx:
                global_used_except |= get_row_diffs(current_rows[i])
                
        curr_row_max = current_rows[row_idx][-1]
        target = curr_row_max - 1
        
        new_row = try_shrink_row(row_idx, target, global_used_except)
        
        if new_row is not None:
            valid, _ = is_valid_row(new_row, global_used_except)
            if valid:
                current_rows[row_idx] = new_row
                new_global_max = get_max_entry(current_rows)
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [list(r) for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count > 100:
            # Perturbation
            pert_idx = random.randint(0, n_rows - 1)
            pert_scope = current_max + 10 
            global_used_except_pert = 0
            for i in range(n_rows):
                if i != pert_idx:
                    global_used_except_pert |= get_row_diffs(current_rows[i])
                        
            new_pert_row = random_row(pert_scope, global_used_except_pert, attempts=10)
            if new_pert_row is not None:
                valid, _ = is_valid_row(new_pert_row, global_used_except_pert)
                if valid:
                    current_rows[pert_idx] = new_pert_row
                    current_max = get_max_entry(current_rows)
                    no_improve_count = 0

    # Final Validation
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    if not validate(best_rows):
        rows_fallback = []
        gu = 0
        for i in range(n_rows):
            r = greedy_row(1000, gu)
            if r is None:
                r = [0] + list(range(1, k+1))
            rows_fallback.append(r)
            gu |= get_row_diffs(r)
        best_rows = rows_fallback

    return best_rows
