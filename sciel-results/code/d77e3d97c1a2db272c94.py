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
    
    def solve_csp(n_rows, k, max_entry, timeout):
        """
        CSP solver using bitset difference tracking and dynamic ordering.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        # Precompute difference masks for quick lookup
        # For a given current_row and candidate x, compute which diffs would be added
        def compute_diff_mask(current_row, x):
            """Compute bitset of all differences x - a for a in current_row."""
            mask = 0
            for a in current_row:
                d = x - a
                if d <= 0:
                    continue
                mask |= (1 << d)
            return mask
        
        def build_row(current_row, used_bits, deadline):
            """
            Recursively build a row using backtracking with bitset checks.
            Uses MRV-like heuristic: try values that leave most flexibility.
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Generate candidate values with pruning
            candidates = []
            for x in range(last + 1, max_entry + 1):
                # Prune: need room for remaining-1 more elements
                if max_entry - x < remaining - 1:
                    break
                
                # Compute new differences as bitset
                new_diff_bits = compute_diff_mask(current_row, x)
                
                # Check if any of these diffs are already used
                if new_diff_bits & used_bits:
                    continue
                
                # Score: prefer values that use fewer small differences
                # Small differences are more constrained
                small_count = 0
                temp = new_diff_bits
                for i in range(1, 16):
                    if (temp >> i) & 1:
                        small_count += 1
                candidates.append((small_count, x, new_diff_bits))
            
            # Sort by score (fewer small diffs = more flexible)
            candidates.sort()
            
            for _, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one with dynamic ordering
        # Try rows in order of "tightness" - rows that are harder to place first
        row_order = list(range(n_rows))
        
        # Heuristic: try to build rows that have fewer available slots first
        # For now, just process in order but with good internal heuristics
        
        for row_idx in row_order:
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                # Update used_bits with all diffs from this row
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
    
    # Strategy: try max_entry values from lb upward with time budgets
    # Use a smarter search: try lb, then lb+1, lb+2, etc. with decreasing budgets
    
    max_search_range = min(greedy_max, lb + 25)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5  # leave 5 sec safety
        
        if remaining_time < 1.0:
            break
        
        # Allocate time: more time for values closer to lb
        # Use a budget that decreases as we go further from lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(20.0, remaining_time)
        elif distance_from_lb <= 3:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 8:
            budget = min(5.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(3.0, remaining_time)
        else:
            budget = min(1.5, remaining_time)
        
        if budget < 0.5:
            break
        
        result = solve_csp(n_rows, k, max_entry, budget)
        if result is not None:
            best_rows = result
            best_max = max_entry
            break  # Found best possible for this max_entry
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                budget = min(remaining_time, 15.0)
                result = solve_csp(n_rows, k, mid, budget)
                if result is not None:
                    best_rows = result
                    best_max = mid
    
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
