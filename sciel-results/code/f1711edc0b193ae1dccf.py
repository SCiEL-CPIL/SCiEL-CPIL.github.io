# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque
from functools import lru_cache

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Bitset-based solver using Python integers as bitsets
    # Bit i is set if difference i is used
    
    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        rng = random.Random(seed)
        
        # We build rows one by one.
        # To improve performance, we can try to build the "hardest" row first?
        # Or just randomize the order of rows?
        # For now, we build in fixed order but randomize the choices within the row.
        
        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Generate candidate values
            # We need to pick x such that x > last and x <= max_entry
            # Also, we must be able to fit 'remaining' more elements after x.
            # So x <= max_entry - remaining + 1
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
                # Compute new differences as bitset
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Heuristic: 
                # 1. Prefer smaller x to keep the row compact? 
                #    But if we pick too small, we might block future rows?
                # 2. Prefer x that uses "rare" differences?
                #    Small differences are more likely to be used by other rows?
                #    Actually, all differences must be distinct.
                #    So using a small difference 'd' consumes 'd'.
                #    If we use many small differences, we might run out of small numbers
                #    for other rows? No, other rows can use large numbers.
                #    But if we want to minimize the GLOBAL max entry, we should try to
                #    pack the differences densely?
                
                # Let's just collect valid candidates and shuffle them.
                candidates.append(x)
            
            if not candidates:
                return False
            
            # Shuffle to add randomness
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return False
                
                # Recompute bits for this x (or store them, but recomputing is cheap)
                new_diff_bits = 0
                for a in current_row:
                    d = x - a
                    new_diff_bits |= (1 << d)
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Try multiple restarts if possible within budget
        # But the function is called with a specific budget.
        # We can do internal restarts if we fail quickly.
        
        # For simplicity, we do one deep search with randomization.
        # If we have time, the outer loop will call this again with different seeds?
        # No, the outer loop iterates over max_entry.
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward
    # lb is the theoretical minimum.
    # We try a range of values.
    
    # If lb is much smaller than greedy_max, we have a lot of room.
    # We try to find the smallest M.
    
    search_upper_bound = greedy_max
    
    # We can also try to jump to a value slightly above lb if lb is very small
    # But for this problem lb is tight (112).
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2  # leave 2 sec safety
        
        if remaining_time < 0.5:
            break
        
        # Allocate time: more time for values closer to lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 10:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Try a few random seeds for each max_entry
        found = False
        # Number of restarts depends on budget
        num_restarts = max(1, int(budget / 1.0))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            # Give each restart a slice of the budget
            restart_budget = budget / (num_restarts - seed) if seed < num_restarts - 1 else budget - seed * (budget/num_restarts)
            # Simpler: just use a fixed small time per restart
            restart_budget = min(1.0, budget / (num_restarts - seed))
            
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Found a solution for this max_entry, which is the best possible for this M
            # Since we are iterating M from lb upwards, this is the global minimum found so far
            # We can break if we want the absolute best, but maybe we can find a smaller M?
            # No, we started from lb. If we found one at M, it's the best.
            break
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
