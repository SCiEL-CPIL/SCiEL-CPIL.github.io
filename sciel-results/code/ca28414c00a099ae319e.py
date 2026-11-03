"""Structural change: CSP backtracking with forward checking + randomized restarts.

We model each row as a Sidon set (all pairwise differences distinct) and require that
differences across all rows are globally distinct. We search for a tight scope M by
binary search, using randomized restarts to escape local minima. A deadline guard
ensures we return the best valid solution found before the 120s limit.
"""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s kill
    total_diffs = n_rows * k * (k + 1) // 2
    # Counting bound: we need at least total_diffs distinct differences in [1, M]
    # So M must satisfy M*(M+1)/2 >= total_diffs
    min_M = 0
    lo, hi = 1, 200
    while lo < hi:
        mid = (lo + hi) // 2
        if mid * (mid + 1) // 2 >= total_diffs:
            hi = mid
        else:
            lo = mid + 1
    min_M = lo
    
    best_solution = None
    best_M = None
    
    # Try scopes from a reasonable lower bound upward
    # Start near the counting bound but leave some slack
    start_M = max(min_M, 100)
    
    # We'll try to find a solution with scope M, incrementing M if needed
    for M in range(start_M, min(start_M + 40, 200)):
        if time.time() > deadline - 5:
            break
        # Try multiple random restarts for this scope
        found = _try_scope(n_rows, k, M, deadline)
        if found is not None:
            best_solution = found
            best_M = M
            break
    
    # If we didn't find anything in the range, use greedy as fallback
    if best_solution is None:
        best_solution = _greedy_fallback(n_rows, k)
    
    return best_solution


def _try_scope(n_rows, k, M, deadline):
    """Try to find n_rows Sidon sets within [0, M] with globally distinct differences.
    Uses CSP backtracking with forward checking and randomized restarts."""
    total_diffs = n_rows * k * (k + 1) // 2
    
    # All possible differences are 1..M
    # We need to assign total_diffs distinct differences to the n_rows rows
    # Each row of k+1 elements starting at 0 has k*(k+1)/2 differences
    
    # Strategy: build rows one at a time. For each row, choose k elements from [1..M]
    # (since 0 is fixed) such that:
    #   - All pairwise differences within the row are distinct (Sidon property)
    #   - None of the differences have been used by previous rows
    
    # Precompute: for a given set of used differences, find valid next elements
    # This is expensive, so we use a smarter approach:
    # Enumerate candidate rows (Sidon sets of size k+1 in [0..M]) and select disjoint ones.
    
    # Actually, let's do incremental backtracking with randomization.
    # For each row, we pick k values from [1..M] in increasing order.
    
    def backtrack(row_idx, used_diffs, rows_so_far):
        if time.time() > deadline - 2:
            return None
        if row_idx == n_rows:
            return rows_so_far[:]
        
        # Choose k values from [1..M] that form a valid Sidon set with unused diffs
        # We'll build the row incrementally
        return _build_row(row_idx, used_diffs, rows_so_far, M, k, deadline)
    
    def _build_row(row_idx, used_diffs, rows_so_far, M, k, deadline):
        """Build one row: choose k values v1 < v2 < ... < vk from [1..M]
        such that all differences vi - vj (i>j) are distinct and not in used_diffs."""
        # Randomized restart: shuffle candidate order
        candidates = list(range(1, M + 1))
        random.shuffle(candidates)
        
        # We need to pick k values. Use backtracking with the shuffled order.
        # State: current partial row (list of chosen values), current set of diffs in this row
        def rec(pos, partial_row, row_diffs):
            if time.time() > deadline - 2:
                return None
            if pos == k:
                # Check all diffs are valid (already checked incrementally)
                new_used = used_diffs | row_diffs
                rows_so_far.append([0] + partial_row)
                result = backtrack(row_idx + 1, new_used, rows_so_far)
                rows_so_far.pop()
                return result
            
            # Try candidates in shuffled order
            # Optimization: skip values that would create a duplicate diff
            for val in candidates:
                # Check: val - v for all v in partial_row must be positive and not in row_diffs and not in used_diffs
                valid = True
                new_diffs = set()
                for v in partial_row:
                    d = val - v
                    if d <= 0:
                        valid = False
                        break
                    if d in row_diffs or d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Also check that new_diffs doesn't have internal duplicates (already ensured by construction since each d is unique per pair)
                # But we need to ensure no two pairs give same diff - that's the Sidon property
                # Actually the above check ensures d not in row_diffs, which covers this.
                
                result = rec(pos + 1, partial_row + [val], row_diffs | new_diffs)
                if result is not None:
                    return result
            
            return None
        
        # Start with empty partial row
        return rec(0, [], set())
    
    # Multiple randomized restarts
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline - 2:
            break
        random.seed(hash((n_rows, k, M, r)) % (2**32))
        result = backtrack(0, set(), [])
        if result is not None:
            return result
    
    return None


def _greedy_fallback(n_rows, k):
    """Original greedy as fallback."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
