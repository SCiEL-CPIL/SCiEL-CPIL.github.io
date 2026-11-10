import time
import random
import sys

# Increase recursion limit for deep searches
sys.setrecursionlimit(10000)

# Global deadline
DEADLINE = 0
TIME_LIMIT = 115  # Keep buffer for safety

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT
    
    # 1. Get a baseline solution using a fast greedy method
    # This ensures we always return a valid result if the search fails or times out
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Attempt to find a better solution via optimized backtracking with increasing M
    # The theoretical lower bound for D(7,5) is 105.
    # We search from 105 up to the baseline scope.
    best_found = baseline_rows
    best_scope = baseline_scope
    
    # Generic counting bound: n_rows * k*(k+1)/2 differences must fit in [1, M]
    generic_lb = n_rows * (k * (k + 1) // 2)
    
    # Specific lower bound for D(7,5) is 105
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)
    
    # If baseline is already very good, we might want to check lower bounds
    # But for D(7,5), 105 is the hard lower bound.
    
    for M in range(start_m, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Try to construct n_rows rows with max element M
        # We use a bitmask-based backtracking solver with randomized restarts
        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            return solution
            
    # If loop finishes or times out, return the best found (baseline)
    return best_found

def greedy_construct(n_rows, k):
    """
    Fast greedy algorithm for baseline.
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

def solve_optimized(n_rows, k, M):
    """
    Attempts to find a valid set of n_rows rows with max element <= M.
    Uses bitmask representation for used differences.
    Includes stronger pruning: checks if enough differences remain for ALL remaining rows.
    Includes randomized restarts to escape deterministic dead-ends.
    """
    diffs_per_row = k * (k + 1) // 2
    
    # We will try a few randomized restarts if the first deterministic one fails
    # Or rather, we integrate randomization into the candidate selection order.
    
    # To allow randomization in the inner loop, we pass a seed or use a global random state
    # Here we use a simple approach: try deterministic first. If it fails, try a few randomized versions.
    
    # However, a single deep DFS with good pruning is often enough.
    # The "randomization" can be applied to the *order* of candidates.
    
    # Let's implement a solver that accepts a 'randomize' flag.
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_row(current_row, current_row_mask, rows_remaining, randomize_order=False):
        """
        Recursively builds the current row.
        current_row: list of elements so far
        current_row_mask: bitmask of differences already present in current_row
        rows_remaining: number of rows left to build AFTER this one
        """
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            # Row is complete
            # Update global used_mask
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        # Pruning 1: Check if enough diffs remain for this row AND all subsequent rows
        # Diffs needed for this row: diffs_per_row
        # Diffs used in this row so far: popcount(current_row_mask)
        # Remaining needed for this row: diffs_per_row - popcount(current_row_mask)
        
        # Total diffs needed for all remaining work:
        # (rows_remaining * diffs_per_row) + (diffs_per_row - popcount(current_row_mask))
        
        # Total diffs available: M - popcount(used_mask)
        
        remaining_needed = (rows_remaining + 1) * diffs_per_row - popcount(current_row_mask)
        available = M - popcount(used_mask)
        
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        if not candidates:
            return False
            
        # Heuristic: 
        # If randomize_order is True, shuffle the candidates.
        # Otherwise, sort by x (smallest first) to encourage tight packing.
        if randomize_order:
            random.shuffle(candidates)
        else:
            # candidates are already generated in increasing x order
            # We can stick with this, or sort by some other metric?
            # Smallest x is generally good for minimizing M.
            pass
            
        for x, new_mask in candidates:
            current_row.append(x)
            
            # Recurse
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1, randomize_order):
                return True
            
            current_row.pop()
            
        return False

    # Strategy:
    # 1. Try deterministic search (randomize=False)
    # 2. If it fails, try a few randomized searches (randomize=True)
    
    # We need to reset state for each attempt
    
    # Attempt 1: Deterministic
    used_mask = 0
    rows = []
    
    # We need to wrap the build_row call to handle the loop over n_rows
    # The build_row function above updates used_mask non-locally.
    # We need a helper to run the full sequence of rows.
    
    def try_solve(randomize):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        
        # We need to define build_row inside try_solve to capture the local used_mask?
        # No, used_mask is nonlocal to build_row, but build_row is defined in solve_optimized.
        # This is tricky with the current structure.
        
        # Let's restructure: pass used_mask and rows as arguments or use a class/closure properly.
        # For performance, nonlocal is fine, but we need to reset it.
        
        # Let's redefine build_row to take used_mask and rows as parameters? 
        # That adds overhead. 
        # Alternative: Keep used_mask and rows in the outer scope of solve_optimized, 
        # but reset them before each attempt.
        
        # The current build_row uses `nonlocal used_mask`. 
        # If we call try_solve, we must ensure used_mask is reset.
        
        # Let's just inline the logic for clarity and speed in the final version.
        
        pass

    # Re-implementing with a cleaner state management
    
    def run_search(randomize):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        
        # Redefine build_row to use the local used_mask/rows? 
        # No, the nonlocal keyword in build_row refers to the scope of solve_optimized.
        # So if we reset used_mask and rows in solve_optimized, build_row will see the new values.
        
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1, randomize):
                return False
        return True

    # First attempt: Deterministic
    if run_search(False):
        return rows
        
    # Second attempt: Randomized (few restarts)
    # If deterministic fails, it might be due to a specific bad path.
    # Randomizing the candidate order might find a solution.
    for _ in range(3):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows
            
    return None

# The code above has a subtle bug: `used_mask` and `rows` are defined in `solve_optimized`.
# `build_row` uses `nonlocal used_mask`.
# `run_search` modifies `used_mask` and `rows` in `solve_optimized` scope.
# This should work.

# However, there is a performance concern: `build_row` is defined once.
# The `randomize` flag is passed.
# This looks correct.

# Let's double check the pruning logic.
# `remaining_needed = (rows_remaining + 1) * diffs_per_row - popcount(current_row_mask)`
# `rows_remaining` is the number of rows AFTER the current one.
# So total rows to complete = current (1) + remaining.
# Total diffs needed = (1 + rows_remaining) * diffs_per_row.
# Diffs already secured in current row = popcount(current_row_mask).
# So yes, the formula is correct.

# One more optimization: 
# If `M` is small, the search space is small.
# If `M` is large, the search space is huge.
# The greedy baseline for D(7,5) usually yields a scope around 130-150.
# The optimal is 112.
# The search from 105 to 112 is the critical part.
# The deterministic search should be quite fast for M=112 if the pruning is strong.

# Let's ensure `greedy_construct` is robust.
# It returns a valid set, so `baseline_rows` is always valid.

# Final check on imports and globals.
import time
import random
import sys

sys.setrecursionlimit(10000)

DEADLINE = 0
TIME_LIMIT = 115

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT
    
    # 1. Baseline
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Search
    lower_bound = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = max(lower_bound, 0)
        
    for M in range(lower_bound, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            return solution
            
    return baseline_rows

def greedy_construct(n_rows, k):
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

def solve_optimized(n_rows, k, M):
    diffs_per_row = k * (k + 1) // 2
    used_mask = 0
    rows = []
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_row(current_row, current_row_mask, rows_remaining, randomize_order):
        if time.time() > DEADLINE:
            return False
            
        if len(current_row) == k + 1:
            rows.append(current_row)
            nonlocal used_mask
            used_mask |= current_row_mask
            return True
            
        # Pruning: Check if enough diffs remain for this row AND all subsequent rows
        remaining_needed = (rows_remaining + 1) * diffs_per_row - popcount(current_row_mask)
        available = M - popcount(used_mask)
        
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                if d > M:
                    valid = False
                    break
                bit = 1 << d
                if (used_mask & bit) or (current_row_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            if valid:
                candidates.append((x, new_mask))
                
        if not candidates:
            return False
            
        if randomize_order:
            random.shuffle(candidates)
            
        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1, randomize_order):
                return True
            current_row.pop()
            
        return False

    def run_search(randomize):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1, randomize):
                return False
        return True

    # 1. Deterministic
    if run_search(False):
        return rows
        
    # 2. Randomized Restarts
    for _ in range(5):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows
            
    return None
