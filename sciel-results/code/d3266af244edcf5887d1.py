"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    Uses a CSP approach with bitset-based difference constraint propagation
    and dynamic variable ordering to find tight solutions.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
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
    # We need differences from 1 to max_entry, so we need max_entry bits
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        import random as rnd
        rng = rnd.Random(seed)
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        def available_diffs(used_bits):
            return max_entry - used_bits.bit_count()

        def build_row(current_row, used_bits, deadline, rng):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Forward Checking: ensure enough differences are available
            needed_diffs = remaining * len(current_row) + (remaining * (remaining - 1)) // 2
            if available_diffs(used_bits) < needed_diffs:
                return False

            candidates = []
            for x in range(last + 1, max_entry + 1):
                if max_entry - x < remaining - 1:
                    break
                
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
                
                # Heuristic: Prefer values that use fewer small differences
                small_mask = new_diff_bits & 0x7FFF # 15 bits
                score = small_mask.bit_count()
                
                # Add noise to randomize order
                noise = rng.random() * 0.5
                candidates.append((score + noise, x, new_diff_bits))
            
            if not candidates:
                return False
            
            candidates.sort()
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline, rng):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline, rng):
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
    
    max_search_range = min(greedy_max, lb + 30)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        
        if remaining_time < 1.0:
            break
        
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(20.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(10.0, remaining_time)
        else:
            budget = min(3.0, remaining_time)
        
        if budget < 0.5:
            break
        
        # Try multiple seeds
        num_seeds = max(1, int(budget / 2.0))
        found = False
        for seed in range(num_seeds):
            if time.time() > start_time + time_limit - 5:
                break
            sub_budget = budget / (num_seeds - seed)
            if sub_budget < 0.5:
                break
                
            result = solve_csp(n_rows, k, max_entry, sub_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                budget = min(remaining_time, 10.0)
                num_seeds = max(1, int(budget / 2.0))
                for seed in range(num_seeds):
                    if time.time() > start_time + time_limit - 5:
                        break
                    sub_budget = budget / (num_seeds - seed)
                    if sub_budget < 0.5:
                        break
                    result = solve_csp(n_rows, k, mid, sub_budget, seed=seed)
                    if result is not None:
                        best_rows = result
                        best_max = mid
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
