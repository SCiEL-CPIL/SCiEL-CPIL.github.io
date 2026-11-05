"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import heapq

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    
    Strategy: Iterative deepening on scope M. For each M, try to find a valid
    assignment using randomized greedy with backtracking. Use bitmask for
    difference tracking for speed.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Counting bound: need at least total_diffs distinct differences in [1, M]
    # So M must satisfy M >= total_diffs (roughly)
    # More precisely, differences range from 1 to M, and we need total_diffs distinct ones
    min_possible_M = total_diffs  # lower bound
    
    # The simple counting bound for (7,5) is 105
    # Best known is 112
    
    # Try scopes starting from a reasonable lower bound
    # For (7,5): total_diffs = 7 * 15 = 105, so M >= 105
    # Try from min_possible_M up to some upper bound
    
    upper_bound = min_possible_M + 30  # reasonable upper bound
    
    # For general cases, be more careful
    if n_rows * n_diffs_per_row > 200:
        upper_bound = min_possible_M + 50
    
    best_rows = None
    best_M = None
    
    # Try each scope M from lower bound upward
    for M in range(min_possible_M, upper_bound + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this scope
        num_restarts = 50
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            rows = _try_construct(n_rows, k, M, restart)
            if rows is not None:
                best_rows = rows
                best_M = M
                found = True
                break
        
        if found:
            break
    
    # If no solution found within the scope range, fall back to greedy
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows


def _try_construct(n_rows, k, M, seed):
    """
    Try to construct n_rows rows with scope M using randomized backtracking.
    Returns list of rows if successful, None otherwise.
    """
    rng = random.Random(seed * 1000003 + 7)
    
    n_diffs_per_row = k * (k + 1) // 2
    
    # Track used differences as a set of booleans (bitmask-like)
    # Differences range from 1 to M
    used = bytearray(M + 1)  # used[d] = 1 if difference d is used
    
    rows = []
    
    def backtrack(row_idx, current_row, depth):
        """
        row_idx: which row we're constructing
        current_row: list of values in current row (starts with [0])
        depth: how many more values to add to current row
        """
        if time.time() > 118.0:  # hard deadline check
            return False
        
        if row_idx == n_rows:
            return True
        
        if depth == 0:
            # Finished current row, move to next
            rows.append(current_row)
            return backtrack(row_idx + 1, [0], k)
        
        # Try to find a value x > current_row[-1] such that all differences
        # x - a for a in current_row are unused
        last_val = current_row[-1]
        
        # Generate candidate values in randomized order for diversity
        # Values must be in range [last_val + 1, M]
        candidates = list(range(last_val + 1, M + 1))
        
        # For efficiency, try smaller values first but with some randomization
        if len(candidates) > 20:
            # Shuffle but keep some bias toward smaller values
            for i in range(len(candidates)):
                j = rng.randint(i, len(candidates) - 1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        
        for x in candidates:
            # Check if all differences x - a are unused
            valid = True
            diffs = []
            for a in current_row:
                d = x - a
                if d <= 0 or d > M:
                    valid = False
                    break
                if used[d]:
                    valid = False
                    break
                diffs.append(d)
            
            if not valid:
                continue
            
            # Mark differences as used
            for d in diffs:
                used[d] = 1
            
            # Recurse
            if backtrack(row_idx, current_row + [x], depth - 1):
                return True
            
            # Undo
            for d in diffs:
                used[d] = 0
        
        return False
    
    # Start backtracking
    success = backtrack(0, [0], k)
    
    if success:
        return rows
    return None


def _greedy_construct(n_rows, k):
    """
    Fallback greedy construction (same as original).
    """
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

# EVOLVE-BLOCK-END
