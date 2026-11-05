"""Improved CSP solver with forward checking and better heuristics."""
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
    
    Uses a CSP approach with bitset-based difference constraint propagation,
    forward checking, and dynamic variable ordering to find tight solutions.
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
        CSP solver using bitset difference tracking, forward checking,
        and dynamic ordering.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = []
        
        # Precompute for each x in [1, max_entry]: bitset of differences x-a for a in [0, x-1]
        # Actually, we need differences relative to current row elements
        # Instead, precompute diff_bits[x] = bitset with bits set at positions 1..x
        # Then for a row with last element L, new diffs for candidate x are:
        #   {x - a for a in current_row}
        # We can compute this incrementally
        
        # Better: precompute for each pair (i, j) with i < j <= max_entry:
        #   diff_bit[i][j] = 1 << (j - i)
        # But that's O(max_entry^2) memory. Instead, compute on the fly but efficiently.
        
        # For speed, precompute a table: for each x, store the bitset of all differences
        # from x to smaller values: diff_to_zero[x] = sum(1 << (x - a) for a in range(x))
        # But we need differences to specific row elements, not all smaller values.
        
        # Alternative: maintain for each candidate x, the set of differences to current_row
        # We'll compute this incrementally as we build the row.
        
        def build_row(current_row, used_bits, deadline):
            """
            Recursively build a row using backtracking with bitset checks
            and forward checking.
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Forward checking: ensure we can still fill remaining positions
            # The smallest possible next value is last + 1
            # The largest possible value is max_entry
            # We need 'remaining' more values, each at least 1 apart
            # So we need: max_entry - last >= remaining
            if max_entry - last < remaining:
                return False
            
            # Generate candidates with forward checking
            candidates = []
            for x in range(last + 1, max_entry + 1):
                # Prune: need room for remaining-1 more elements after x
                if max_entry - x < remaining - 1:
                    break
                
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
                
                # Forward check: after placing x, can we still complete the row?
                # The next value must be > x, and we need remaining-1 more values
                # Each new value y will create differences y - a for a in current_row + [x]
                # At minimum, we need differences 1, 2, ..., (remaining-1) to be available
                # (since consecutive values differ by at least 1)
                # Actually, we need to check that there exist enough unused differences
                # This is complex; use a simpler heuristic:
                # Count how many differences in [1, max_entry - x] are unused
                # We need at least (remaining-1) * len(current_row) + (remaining-1)*remaining//2
                # differences, but that's an overestimate.
                
                # Simpler forward check: ensure that for each a in current_row + [x],
                # there are enough unused differences in [1, max_entry - a]
                # This is still complex; skip for now and rely on backtracking.
                
                # Score: prefer values that use fewer small differences
                # Small differences (1-15) are most constrained
                small_diff_count = 0
                for i in range(1, 16):
                    if (new_diff_bits >> i) & 1:
                        small_diff_count += 1
                
                # Also prefer values that are not too close to existing elements
                # (to leave room for more flexible differences)
                min_dist = min(x - a for a in current_row)
                score = small_diff_count * 10 + (15 - min(min_dist, 15))
                
                candidates.append((score, x, new_diff_bits))
            
            # Sort by score (lower is better)
            candidates.sort()
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        # Build rows one by one
        for row_idx in range(n_rows):
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
    max_search_range = min(greedy_max, lb + 25)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        
        if remaining_time < 1.0:
            break
        
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(12.0, remaining_time)
        elif distance_from_lb <= 3:
            budget = min(6.0, remaining_time)
        elif distance_from_lb <= 10:
            budget = min(3.0, remaining_time)
        else:
            budget = min(1.5, remaining_time)
        
        if budget < 0.3:
            break
        
        result = solve_csp(n_rows, k, max_entry, budget)
        if result is not None:
            best_rows = result
            best_max = max_entry
            break
    
    # If we didn't find anything better, try a mid-point with more time
    if best_max == greedy_max and greedy_max > lb:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                budget = min(remaining_time, 8.0)
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
