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
    
    # 1. Baseline via greedy
    baseline_rows = greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline_rows) if baseline_rows else 0
    
    # 2. Search for better solution
    # Lower bound for D(7,5) is 105
    generic_lb = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb
        
    start_m = max(lower_bound, generic_lb)
    
    # We search from lower_bound up to baseline_scope
    # If we find a solution at M, it's the optimal for that M (since we go up)
    # But we want the smallest M possible.
    # The loop below tries M from start_m upwards.
    # If solve_optimized returns a solution, that M is the answer.
    
    for M in range(start_m, baseline_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Try to construct n_rows rows with max element M
        solution = solve_optimized(n_rows, k, M)
        if solution is not None:
            return solution
            
    # If loop finishes or times out, return the best found (baseline)
    return baseline_rows

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
    
    # State variables
    used_mask = 0
    rows = []
    
    def popcount(x):
        return bin(x).count('1')
        
    def build_row(current_row, current_row_mask, rows_remaining, randomize_order):
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
            nonlocal used_mask
            used_mask |= current_row_mask
            rows.append(current_row)
            return True
            
        # Pruning 1: Check if enough diffs remain for this row AND all subsequent rows
        remaining_needed = (rows_remaining + 1) * diffs_per_row - popcount(current_row_mask)
        available = M - popcount(used_mask)
        
        if available < remaining_needed:
            return False
            
        last_val = current_row[-1]
        candidates = []
        
        # Optimization: Pre-compute bit checks
        # For each candidate x, we need to check if (x - a) bits are free in used_mask and current_row_mask
        # We can iterate x from last_val+1 to M
        
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                # d is always <= M because x <= M and a >= 0, so d <= M. 
                # Wait, if a=0, d=x. x<=M. So d<=M is always true.
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
            pass
            
        for x, new_mask in candidates:
            current_row.append(x)
            
            # Recurse
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
    # If deterministic fails, it might be due to a specific bad path.
    # Randomizing the candidate order might find a solution.
    # We limit the number of restarts to avoid wasting time if M is too small.
    for _ in range(10):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows
            
    return None
