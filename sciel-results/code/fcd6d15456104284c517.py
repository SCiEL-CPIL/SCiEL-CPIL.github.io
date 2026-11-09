import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not diffs & used:
                    row.append(x)
                    used.update(diffs)
            rows.append(row)
        return rows

    def is_valid(rows):
        seen = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    # Start with greedy baseline
    current_rows = greedy_construct()
    current_scope = get_scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope

    # Local search: try to swap elements within a row or across rows
    # to reduce the scope while maintaining validity
    
    def swap_within_row(rows, row_idx, pos_a, pos_b):
        """Swap elements at pos_a and pos_b within row row_idx, maintaining sorted order"""
        new_rows = [list(r) for r in rows]
        row = new_rows[row_idx]
        # We can only swap if the result is still sorted
        # Since we're swapping positions, we need to be careful
        # Actually, let's just try generating new rows by modifying one element at a time
        
    # Better approach: try to replace one row at a time with a better one
    # A row is better if it uses smaller max value while maintaining global validity
    
    # Simulated annealing approach
    # State: the set of rows
    # Move: replace one row with a new valid row (given the used differences from other rows)
    # Cost: scope
    
    def rebuild_row_given_forbidden(forbidden_diffs, max_val, seed=None):
        """
        Try to build a row of length k+1 starting at 0 with all entries <= max_val
        such that none of its differences are in forbidden_diffs.
        Uses randomized backtracking.
        """
        if seed is not None:
            random.seed(seed)
        
        def backtrack(current_row):
            if len(current_row) == k + 1:
                return current_row[:]
            
            last = current_row[-1]
            # Try candidates from last+1 to max_val
            # Use randomization to explore different parts of the space
            candidates = []
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if not (new_diffs & forbidden_diffs):
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Shuffle to randomize
            random.shuffle(candidates)
            
            # Try a few candidates to avoid deep backtracking
            for x in candidates[:10]:
                new_diffs = {x - a for a in current_row}
                current_row.append(x)
                result = backtrack(current_row)
                current_row.pop()
                if result is not None:
                    return result
            
            return None
        
        return backtrack([0])
    
    # Main search loop
    while time.time() < deadline - 5:
        # Try to reduce scope
        target_scope = current_scope - 1
        
        if target_scope < 0:
            break
            
        # Get forbidden differences (all diffs except from one row at a time)
        all_diffs_by_row = []
        for i, row in enumerate(current_rows):
            row_diffs = set()
            for a in range(len(row)):
                for b in range(a+1, len(row)):
                    row_diffs.add(row[b] - row[a])
            all_diffs_by_row.append(row_diffs)
        
        found_improvement = False
        for row_idx in range(n_rows):
            # Get forbidden diffs (from all other rows)
            forbidden = set()
            for j in range(n_rows):
                if j != row_idx:
                    forbidden.update(all_diffs_by_row[j])
            
            # Try to rebuild row row_idx with smaller scope
            # Try a few times with different random seeds
            for attempt in range(50):
                if time.time() > deadline - 3:
                    break
                    
                new_row = rebuild_row_given_forbidden(forbidden, target_scope, seed=attempt*1000+row_idx)
                
                if new_row is not None:
                    # Check if this new row actually has max <= target_scope
                    if max(new_row) <= target_scope:
                        # Try replacing this row
                        new_rows = [list(r) for r in current_rows]
                        new_rows[row_idx] = new_row
                        
                        if is_valid(new_rows):
                            new_scope = get_scope(new_rows)
                            if new_scope < current_scope:
                                current_rows = new_rows
                                current_scope = new_scope
                                best_rows = [list(r) for r in current_rows]
                                best_scope = current_scope
                                found_improvement = True
                                # Recompute all_diffs_by_row
                                all_diffs_by_row = []
                                for i, row in enumerate(current_rows):
                                    row_diffs = set()
                                    for a in range(len(row)):
                                        for b in range(a+1, len(row)):
                                            row_diffs.add(row[b] - row[a])
                                    all_diffs_by_row.append(row_diffs)
                                break  # Break out of attempt loop
                            else:
                                # Even though max(new_row) <= target_scope, the overall scope didn't improve
                                # This means some other row has the max
                                # Try to improve other rows
                                pass
            
            if found_improvement:
                break
        
        if not found_improvement:
            # No improvement found, try with a smaller reduction
            # Or try randomized restarts
            # Let's try a more aggressive random restart
            if random.random() < 0.1:  # Occasionally do a full restart
                # Try to construct from scratch with a tighter bound
                # Use the current scope as an upper bound for the search
                for attempt in range(20):
                    if time.time() > deadline - 3:
                        break
                    
                    # Randomized greedy with the current scope as limit
                    used = set()
                    new_rows = []
                    success = True
                    
                    for _ in range(n_rows):
                        row = [0]
                        while len(row) < k + 1:
                            candidates = []
                            for x in range(row[-1] + 1, current_scope + 1):
                                diffs = {x - a for a in row}
                                if not (diffs & used):
                                    candidates.append(x)
                            
                            if not candidates:
                                success = False
                                break
                            
                            # Pick randomly with bias towards smaller values
                            window = min(3, len(candidates))
                            next_x = random.choice(candidates[:window])
                            row.append(next_x)
                            used.update({next_x - a for a in row[:-1]})
                        
                        if not success:
                            break
                        new_rows.append(row)
                    
                    if success and is_valid(new_rows):
                        new_scope = get_scope(new_rows)
                        if new_scope < current_scope:
                            current_rows = new_rows
                            current_scope = new_scope
                            best_rows = [list(r) for r in current_rows]
                            best_scope = current_scope
                            found_improvement = True
                            break
        
        # If we've been at the same scope for a while, try harder
        # The above loop will naturally try to reduce scope by 1 each time

    return best_rows
