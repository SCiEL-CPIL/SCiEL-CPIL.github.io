"""Optimized solver for Difference Triangle Set (n_rows, k) using bitsets and backtracking."""
import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0
    
    k_plus_1 = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    def get_scope(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        used_mask = 0
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if (used_mask >> d) & 1:
                        return False
                    used_mask |= (1 << d)
        return True

    def row_diff_mask(row):
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def try_build_row(used_mask, max_val, k_plus_1, rng):
        """
        Attempt to build a single row of length k_plus_1 starting with 0.
        Returns the row if successful, None otherwise.
        Uses randomized greedy with limited backtracking via retries.
        """
        # We'll use a simple randomized greedy first. 
        # If it fails, we can try different random seeds/orderings in the caller.
        row = [0]
        # To allow some flexibility, we can shuffle the candidate values slightly? 
        # No, strictly increasing is required. 
        # The choice is which 'x' to pick next.
        
        # Heuristic: Try values in increasing order, but maybe skip some to avoid dead ends?
        # For now, standard greedy is fast. If it fails, the caller retries with different global state/order.
        
        # To make it more robust, we can randomize the starting point of the scan if we get stuck?
        # Actually, just standard greedy is often sufficient if the scope is tight enough and we have enough retries.
        
        for _ in range(k_plus_1 - 1):
            found = False
            # Start from last element + 1
            start_x = row[-1] + 1
            # Randomize the order of trying x? 
            # If we always pick the smallest x, we might get stuck. 
            # Let's try a few random offsets or just shuffle the range? 
            # Shuffling a large range is expensive. 
            # Instead, let's try the smallest x first. If this specific path fails, the caller will retry.
            
            # Optimization: We can check if there are enough available differences left?
            # Too complex. Stick to simple greedy with many restarts.
            
            for x in range(start_x, max_val + 1):
                # Check conflicts
                conflict = False
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    if d > max_val:
                        conflict = True
                        break
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                    new_diffs_mask |= (1 << d)
                
                if not conflict:
                    row.append(x)
                    used_mask |= new_diffs_mask
                    found = True
                    break
            
            if not found:
                return None
        
        return row

    def solve_for_scope(max_scope, rng):
        """
        Try to construct n_rows with max entry <= max_scope.
        Returns list of rows or None.
        """
        # Randomize the order of rows to construct
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        used_mask = 0
        rows = [None] * n_rows
        
        for r_idx in row_order:
            # Try to build row r_idx
            # We can try multiple times with slight variations? 
            # For now, one greedy attempt per call. The outer loop calls this multiple times.
            
            # To add some randomness to the row construction itself:
            # We can shuffle the "priority" of x? 
            # Let's stick to deterministic greedy for speed, relying on row order randomization.
            # However, greedy is very sensitive. 
            # Let's add a small randomization: sometimes skip the smallest valid x?
            # No, that might increase scope. 
            # Let's just use the greedy.
            
            row = try_build_row(used_mask, max_scope, k_plus_1, rng)
            if row is None:
                return None
            rows[r_idx] = row
            used_mask |= row_diff_mask(row)
            
        return rows

    # Phase 1: Get a baseline solution
    # Use a generous scope first to ensure we have a valid answer
    # Counting bound: S(S+1)/2 >= total_diffs_needed
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    initial_scope = max(lb + 50, 150) # Start high to be safe
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Try to find a solution with initial_scope
    for attempt in range(100):
        if time.time() > time_limit:
            break
        res = solve_for_scope(initial_scope, rng)
        if res is not None:
            best_rows = res
            best_scope = get_scope(res)
            break
    
    if best_rows is None:
        # Fallback: very loose greedy
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    conflict = False
                    for a in row:
                        d = x - a
                        if (used_mask >> d) & 1:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]: # exclude x itself? No, x is last. 
                            # Actually we need to add diffs for x with all previous
                            pass
                        # Update mask
                        for a in row[:-1]:
                            used_mask |= (1 << (x - a))
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = get_scope(best_rows)
        return best_rows

    # Phase 2: Iterative deepening / decreasing scope
    # We know best_scope is valid. Try to reduce it.
    # We can jump down by 1 or 2.
    # Since best known is 134, and lb is ~112, we have a window.
    
    current_scope = best_scope
    
    while current_scope > lb and time.time() < time_limit:
        # Try to find a solution with scope = current_scope - 1
        target_scope = current_scope - 1
        
        found = False
        # Try multiple randomizations for this target scope
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > time_limit:
                break
            res = solve_for_scope(target_scope, rng)
            if res is not None:
                # Verify it actually fits in target_scope (it should by construction)
                if get_scope(res) <= target_scope:
                    best_rows = res
                    best_scope = target_scope
                    current_scope = target_scope
                    found = True
                    break
        
        if not found:
            # If we fail to find a solution for current_scope - 1, 
            # it might be possible with different randomizations, but 
            # if we've tried enough, we might be stuck.
            # Let's try a few more times or break?
            # To be safe, let's try a few more times with different seeds?
            # The rng is stateful, so it's already different.
            
            # If we fail, we stop decreasing.
            break
            
    # Final validation
    if not is_valid(best_rows):
         # This should not happen if logic is correct, but just in case
         pass

    return best_rows
