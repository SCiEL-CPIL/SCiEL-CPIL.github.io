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
    
    Uses a hybrid approach:
    1. Try to find a solution with a tight upper bound using backtracking with
       bitset-based difference tracking and heuristic ordering.
    2. If that fails within time limit, fall back to a greedy construction.
    
    The key insight is that we can use a bitset to track which differences
    are already used, and for each row, we can precompute which differences
    would conflict. We use a depth-first search with dynamic variable ordering
    (fill rows with tightest remaining capacity first).
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Start with a reasonable upper bound
    # The simple counting bound: we need n_rows * k*(k+1)/2 distinct differences
    # The largest entry m gives differences up to m, so m*(m+1)/2 >= n_rows * k*(k+1)/2
    # So m >= sqrt(n_rows * k*(k+1)) roughly
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound on max entry
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    # Try to find a solution with max entry close to the lower bound
    # We'll try increasing upper bounds starting from a reasonable estimate
    
    # For small cases, we can be more aggressive
    # The greedy approach typically gives a solution, but not optimal
    # Let's first get a greedy solution to have a baseline
    
    def greedy_construct(n_rows, k):
        """Greedy ruler construction - same as seed but returns the max entry."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Get greedy baseline
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    # Now try to do better with backtracking
    # We'll try to find a solution with max entry < greedy_max
    
    # Time limit: 115 seconds (leave 5 sec for safety)
    start_time = time.time()
    time_limit = 115.0
    
    # For the backtracking approach, we'll use a smarter strategy:
    # Build rows one at a time, and for each row, use a recursive search
    # with the constraint that no difference is reused.
    
    # Key data structure: a set of used differences
    # For each row, we need to choose k+1 positions including 0
    # such that all pairwise differences are not in used_set
    
    # Let's implement a backtracking solver that builds rows one by one
    # For each row, we use a DFS to find a valid set of k+1 positions
    
    def backtrack_solve(n_rows, k, max_entry, timeout):
        """
        Try to find n_rows valid rows with all entries <= max_entry.
        Returns list of rows or None if not found within timeout.
        """
        used_diffs = set()
        rows = []
        deadline = time.time() + timeout
        
        # Precompute all valid differences for a row of length k+1
        # given a set of used differences
        # We'll use a recursive approach to build each row
        
        def build_row(row_idx, current_row, used_diffs, deadline):
            """
            Recursively build a row. current_row is a list starting with [0].
            Returns True if successful, with current_row extended to length k+1.
            """
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            # Try next values from last+1 up to max_entry
            # But we can prune: we need k+1-len(current_row) more elements
            # The smallest possible next value is last+1
            # The largest possible next value is max_entry
            # We need to be able to fit (k+1-len(current_row))-1 more after this
            
            remaining = k + 1 - len(current_row)
            
            # Heuristic: try values that allow the most flexibility
            # Try smaller values first to keep the row compact
            for x in range(last + 1, max_entry + 1):
                if time.time() > deadline:
                    return False
                
                # Check if adding x creates any conflict
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                
                # Also check: can we still complete the row?
                # We need 'remaining - 1' more elements after x
                # The minimum span needed is (remaining - 1) * 1 = remaining - 1
                # So we need max_entry - x >= remaining - 1
                if max_entry - x < remaining - 1:
                    break  # No point trying larger x either
                
                # Try adding x
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                if build_row(row_idx, current_row, used_diffs, deadline):
                    return True
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Build rows one by one
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(row_idx, current_row, used_diffs, deadline):
                rows.append(current_row)
            else:
                return None
        
        return rows
    
    # Try to find a better solution
    # Start from the lower bound and increase
    # But we want to be efficient, so let's try a few candidate max_entries
    
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Try max entries from lb to greedy_max - 1
    # But doing full backtracking for each might be too slow
    # Let's try a few strategic values
    
    # For small n_rows and k, we can try more aggressively
    # For larger cases, we'll be more conservative
    
    # Strategy: try a few max_entry values, with decreasing time budgets
    # Start from the most promising (closest to lb)
    
    # Actually, let's try a different approach: use a single backtracking
    # that tries to minimize the max entry by trying smaller max values first
    
    # Given the time constraints, let's try a limited number of max_entry values
    # with backtracking, each with a time budget
    
    # For very small cases (n_rows * k <= 20), we can be more thorough
    if n_rows * k <= 20:
        # Try to find optimal or near-optimal
        for max_entry in range(lb, min(greedy_max, lb + 50)):
            if time.time() - start_time > time_limit - 5:
                break
            timeout_for_this = min(5.0, time_limit - (time.time() - start_time) - 5)
            if timeout_for_this < 0.5:
                break
            
            result = backtrack_solve(n_rows, k, max_entry, timeout_for_this)
            if result is not None:
                best_rows = result
                best_max = max_entry
                break  # Found a solution with this max, which is better
    else:
        # For larger cases, just try a couple of values
        # Try the lower bound first with a short timeout
        if time.time() - start_time < time_limit - 10:
            timeout_for_this = min(10.0, time_limit - (time.time() - start_time) - 5)
            if timeout_for_this > 1.0:
                result = backtrack_solve(n_rows, k, lb, timeout_for_this)
                if result is not None:
                    best_rows = result
                    best_max = lb
        
        # Try a value between lb and greedy_max
        if best_max == greedy_max and time.time() - start_time < time_limit - 5:
            mid = (lb + greedy_max) // 2
            if mid > lb:
                timeout_for_this = min(5.0, time_limit - (time.time() - start_time) - 5)
                if timeout_for_this > 0.5:
                    result = backtrack_solve(n_rows, k, mid, timeout_for_this)
                    if result is not None:
                        best_rows = result
                        best_max = mid
    
    # Validate the result
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
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
