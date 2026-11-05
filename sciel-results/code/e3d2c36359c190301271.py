# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

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
        CSP solver using bitset difference tracking and randomized backtracking with lookahead.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        rng = random.Random(seed)
        
        # Precompute powers of 2 for faster bit operations? 
        # 1 << d is fast enough.
        
        def count_valid_next(current_row, last, used_bits, max_entry):
            """
            Heuristic: Count how many valid candidates exist for the next element.
            Returns the count.
            """
            remaining = k + 1 - len(current_row)
            if remaining == 0:
                return 1 # Success
            
            min_x = last + 1
            # We need 'remaining' elements total, so 'remaining-1' after this one.
            # So this one must be <= max_entry - (remaining - 1)
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return 0
            
            count = 0
            # Optimization: if range is large, sample? 
            # For k=7, max_entry~134, range is small.
            for x in range(min_x, max_x + 1):
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    count += 1
            return count

        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
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
                
                candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Sort candidates by "flexibility" (number of valid next steps)
            # If we pick an x that leaves many options for the next step, we are less likely to deadlock.
            # This is a "Least Constraining Value" heuristic.
            
            # To save time, we only do full lookahead if the number of candidates is small 
            # or if we are deep in the recursion. 
            # For now, let's do a quick score for all.
            
            scored_candidates = []
            for x, bits in candidates:
                # Lookahead: how many valid choices for the NEXT element?
                # We simulate adding x
                temp_row = current_row + [x]
                temp_used = used_bits | bits
                # Count valid next steps
                # We can reuse count_valid_next but it's a bit slow.
                # Let's just count directly here for speed since we already have bits.
                next_remaining = remaining - 1
                if next_remaining == 0:
                    score = 1
                else:
                    min_y = x + 1
                    max_y = max_entry - (next_remaining - 1)
                    count = 0
                    # Quick check
                    for y in range(min_y, max_y + 1):
                        ok = True
                        for a in temp_row:
                            d = y - a
                            if (temp_used >> d) & 1:
                                ok = False
                                break
                        if ok:
                            count += 1
                    score = count
                
                scored_candidates.append((score, x, bits))
            
            # Sort by score descending (most flexible first)
            # Add some randomness to break ties or explore different branches
            scored_candidates.sort(key=lambda t: t[0], reverse=True)
            
            # Randomize order within same score or just shuffle if scores are equal?
            # Let's shuffle the whole list but bias towards high scores?
            # Simple approach: Shuffle, then sort by score? No, that loses the order.
            # Let's just iterate from highest score to lowest. If we get stuck, we backtrack.
            
            for score, x, bits in scored_candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
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
    
    search_upper_bound = greedy_max
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2  # leave 2 sec safety
        
        if remaining_time < 0.5:
            break
        
        # Allocate time: more time for values closer to lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 10:
            budget = min(8.0, remaining_time)
        else:
            budget = min(4.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Try a few random seeds for each max_entry
        found = False
        # Number of restarts depends on budget
        # The lookahead heuristic is expensive, so we might need fewer restarts
        num_restarts = max(1, int(budget / 2.0))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            restart_budget = budget / (num_restarts - seed) if seed < num_restarts - 1 else budget - seed * (budget/num_restarts)
            restart_budget = min(2.0, restart_budget)
            
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
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
