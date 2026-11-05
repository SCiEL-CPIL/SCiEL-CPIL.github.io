"""Program for difference triangle set (7,5) using maximum weight independent set search."""

import sys
from itertools import combinations
from functools import lru_cache
import time

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    
    Uses a branch-and-bound approach with backtracking and pruning.
    """
    # We need to find rows with minimal scope.
    # For (7, 5): we need 7 rows of 6 integers each starting at 0.
    # Each row has C(6,2) = 15 differences. Total 7*15 = 105 distinct differences needed.
    # The largest difference in a row with scope m is m, so we need at least 105 distinct values.
    
    # Start with the simple counting bound: need at least 105 differences, 
    # so scope >= 105. But known best is 112.
    
    # Strategy: Try scopes from lower bound upward, and for each scope,
    # try to find n_rows valid rows using backtracking with pruning.
    
    # First, compute the lower bound on scope
    # We need n_rows * C(k+1, 2) distinct positive differences
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Lower bound: the largest difference is scope, so we need at least total_diffs_needed
    # distinct values from 1 to scope, hence scope >= total_diffs_needed
    lower_bound = total_diffs_needed
    
    # Try scopes from lower_bound upward
    for scope in range(lower_bound, lower_bound + 200):
        result = _find_rows(n_rows, k, scope)
        if result is not None:
            return result
    
    # Fallback: greedy construction (should not be reached for reasonable inputs)
    return _greedy_construct(n_rows, k)


def _greedy_construct(n_rows, k):
    """Simple greedy fallback."""
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


def _get_row_diffs(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _find_rows(n_rows, k, scope):
    """
    Try to find n_rows valid rows with all entries <= scope.
    Uses backtracking with pruning.
    """
    # Precompute all valid rows with entries in [0, scope] starting at 0
    # A row is a list of k+1 strictly increasing integers starting at 0, all <= scope
    
    # Generate candidate rows. To keep this tractable, we'll use a smarter approach.
    # Instead of enumerating all rows, we build them incrementally with pruning.
    
    # State: used_diffs (set of differences used so far), current rows list
    # We'll use recursive backtracking with memoization-like pruning.
    
    # For efficiency, we'll represent the problem as:
    # At each step, choose a row (list of k+1 ints starting at 0, <= scope)
    # such that its differences don't conflict with used_diffs.
    
    # To make this feasible, we need to be smart about which rows to try.
    # We'll use a priority/ordered search: try rows that use "small" differences first
    # to leave room for larger differences later, or vice versa.
    
    # Actually, let's use a different approach: since the scope is small (<= ~120),
    # we can enumerate all possible rows and then do maximum independent set.
    
    # But the number of possible rows is C(scope, k) which for scope=112, k=5 is C(112,5) ~ 78 million.
    # That's too many.
    
    # Alternative: use backtracking that builds rows one at a time, and within each row,
    # builds the positions incrementally.
    
    return _backtrack_find(n_rows, k, scope, set(), [])


def _backtrack_find(n_rows, k, scope, used_diffs, rows):
    """
    Backtracking search to find n_rows valid rows.
    """
    if len(rows) == n_rows:
        return rows
    
    # Pruning: if we can't possibly fill the remaining rows, return None
    remaining_rows = n_rows - len(rows)
    diffs_per_row = (k + 1) * k // 2
    total_remaining_diffs = remaining_rows * diffs_per_row
    available_diffs = set(range(1, scope + 1)) - used_diffs
    if len(available_diffs) < total_remaining_diffs:
        return None
    
    # Try to build the next row
    # A row is [0, a1, a2, ..., ak] where 0 < a1 < a2 < ... < ak <= scope
    # We'll build it incrementally.
    
    def build_row(position, current_row, row_diffs):
        """Recursively build a row starting from current position."""
        if len(current_row) == k + 1:
            # Check if this row's differences are all unused
            if row_diffs & used_diffs:
                return None
            # Try to continue with this row
            new_used = used_diffs | row_diffs
            new_rows = rows + [current_row]
            result = _backtrack_find(n_rows, k, scope, new_used, new_rows)
            return result
        
        start = current_row[-1] + 1
        # Optimization: we need enough remaining positions to fill the row
        # and enough available differences
        
        for next_val in range(start, scope + 1):
            # Compute new differences added by adding next_val
            new_diffs = {next_val - a for a in current_row}
            
            # Pruning: if any of these diffs are already used or in row_diffs, skip
            if new_diffs & (used_diffs | row_diffs):
                continue
            
            new_row_diffs = row_diffs | new_diffs
            new_current_row = current_row + [next_val]
            
            # Additional pruning: check if we can still complete the row
            # We need (k + 1 - len(new_current_row)) more positions
            remaining_positions = k + 1 - len(new_current_row)
            
            # The minimum possible new differences when adding remaining positions
            # is at least remaining_positions (each new position adds at least 1 new diff)
            # But more precisely, each new position adds (len(new_current_row)) new diffs
            # This is getting complex; let's just proceed and rely on the main pruning.
            
            result = build_row(position + 1, new_current_row, new_row_diffs)
            if result is not None:
                return result
        
        return None
    
    # Start building a row from [0]
    return build_row(0, [0], set())


# The above backtracking is too slow for the given constraints. Let's use a different approach.
# We'll use a randomized/greedy approach with local search, or a more structured search.

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    lower_bound = total_diffs_needed
    
    # Try scopes from lower_bound upward with a time limit
    start_time = time.time()
    max_time = 110  # Leave some margin for the 120s limit
    
    for scope in range(lower_bound, lower_bound + 300):
        if time.time() - start_time > max_time:
            break
        result = _find_rows_efficient(n_rows, k, scope)
        if result is not None:
            return result
    
    # Fallback: greedy construction
    return _greedy_construct(n_rows, k)


def _find_rows_efficient(n_rows, k, scope):
    """
    Efficient search using randomized greedy with restarts.
    """
    import random
    
    # Try multiple random orderings of candidate rows
    for attempt in range(500):
        result = _randomized_greedy(n_rows, k, scope, attempt)
        if result is not None:
            return result
    
    return None


def _randomized_greedy(n_rows, k, scope, seed):
    """
    Randomized greedy: try to build rows one by one, choosing from valid candidates.
    """
    import random
    rng = random.Random(seed)
    
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        # Generate a random valid row
        row = _generate_random_row(k, scope, used_diffs, rng)
        if row is None:
            return None
        
        row_diffs = _get_row_diffs(row)
        if row_diffs & used_diffs:
            return None
        
        used_diffs |= row_diffs
        rows.append(row)
    
    return rows


def _generate_random_row(k, scope, used_diffs, rng):
    """Generate a random row that doesn't conflict with used_diffs."""
    max_attempts = 1000
    for _ in range(max_attempts):
        # Generate random strictly increasing sequence starting at 0
        candidates = list(range(1, scope + 1))
        rng.shuffle(candidates)
        
        # Try to pick k values that form a valid row
        row = [0]
        row_diffs = set()
        
        for i in range(k):
            # Find a value that doesn't create conflicts
            found = False
            for val in candidates:
                if val <= row[-1]:
                    continue
                new_diffs = {val - a for a in row}
                if new_diffs & (used_diffs | row_diffs):
                    continue
                row.append(val)
                row_diffs |= new_diffs
                found = True
                break
            if not found:
                return None
        
        return row
    
    return None
