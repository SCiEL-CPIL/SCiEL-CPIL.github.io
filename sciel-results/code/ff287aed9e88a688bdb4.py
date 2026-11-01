import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on M
    M_lower = total_diffs_needed
    
    # Get an initial upper bound from a fast greedy construction
    greedy_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(greedy_rows)
    best_rows = greedy_rows
    
    # Strategy: Try to fit into smaller M using a structured backtracking search.
    # We iterate M from M_lower up to best_M - 1.
    # For each M, we use a beam search / backtracking with constraint propagation.
    
    # To speed up, we precompute valid rows for a given M? No, C(M, k) is too large.
    # Instead, we generate rows on the fly but with strong heuristics.
    
    # Key Structural Change: 
    # 1. Use a "Most Constrained Variable" heuristic for row ordering? 
    #    Actually, rows are symmetric. But the *differences* are not.
    # 2. Use a Bitmask for used differences if M is small enough? 
    #    M is around 100-120. Bitmask of 120 bits fits in Python int.
    #    This makes conflict checking O(1) bitwise AND instead of set intersection.
    
    # Let's try M from max(M_lower, best_M - 10) up to best_M - 1.
    # If we find a solution for a smaller M, we return it.
    
    search_start_M = max(M_lower, best_M - 15)
    
    for M in range(search_start_M, best_M):
        if time.time() > deadline - 5:
            break
            
        # Try to solve for this M
        solution = _solve_packing_bitmask(n_rows, k, M, deadline)
        if solution is not None:
            best_rows = solution
            best_M = M
            # Since we go from low to high, the first valid M is optimal for this search range.
            # But we might have skipped some M if search_start_M > M_lower.
            # If we found a solution at M, it's better than best_M.
            # We can break if we want to save time, but maybe we can go lower?
            # No, we iterated from low to high. So this is the best found so far.
            # We can continue to see if we can find an even smaller one? 
            # No, we started at search_start_M. If search_start_M > M_lower, we missed [M_lower, search_start_M).
            # Let's just return the first one found in this range, or continue to M_lower?
            # Given time constraints, finding one is good.
            # Actually, let's just break and return this improvement.
            # Wait, the loop goes from search_start_M to best_M-1.
            # If M_lower < search_start_M, we should ideally check those too.
            # But checking all M from M_lower might be too slow.
            # Let's stick to the range we searched.
            break

    # If no improvement found in the range, return the greedy best
    return best_rows

def _solve_packing_bitmask(n_rows, k, M, deadline):
    """
    Solve the packing problem for a specific M using bitmask-based conflict checking.
    Returns a list of rows or None.
    """
    # Precompute all valid difference masks for a single row? No, too many.
    # Instead, generate rows on the fly.
    
    # Heuristic: Sort the rows we generate? No, rows are generated sequentially.
    # To improve success rate, we can try multiple restarts with different random seeds.
    
    # Bitmask of available differences: bits 1..M are set.
    # diff d corresponds to bit (d-1) or bit d? Let's use bit d for difference d.
    # M is up to ~120. Python ints are arbitrary precision, so this is fine.
    full_mask = (1 << (M + 1)) - 2  # Bits 1 to M set
    
    used_mask = 0
    rows = []
    
    def backtrack(idx):
        nonlocal used_mask
        if time.time() > deadline - 2:
            return False
        
        if idx == n_rows:
            return True
        
        # Generate candidate rows for this position
        # Heuristic: Try rows that use "rare" or "central" differences first?
        # Randomization helps avoid local traps.
        
        # To make it more effective, we can shuffle the order of attempts.
        # We'll try a few random rows. If we fail, we backtrack.
        
        # Optimization: If we are at the last row, we just need ANY valid row.
        # For earlier rows, we want to leave "flexible" differences for later.
        
        # Simple strategy: Try random valid rows.
        # To speed up, we can pre-check if enough differences are left.
        # remaining_diffs = (full_mask ^ used_mask).bit_count()
        # if remaining_diffs < (n_rows - idx) * diffs_per_row: return False
        
        # Let's try up to 1000 random rows for this position.
        for _ in range(1000):
            if time.time() > deadline - 2:
                return False
            
            # Generate a random valid row
            row = _generate_random_row(k, M, used_mask)
            if row is None:
                continue
                
            # Calculate mask for this row
            row_mask = 0
            valid = True
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    row_mask |= bit
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Conflict check with current used_mask is already done in generation?
            # _generate_random_row checks against used_mask passed to it.
            # So row_mask & used_mask should be 0.
            
            # Accept and recurse
            old_used = used_mask
            used_mask |= row_mask
            rows.append(row)
            
            if backtrack(idx + 1):
                return True
            
            rows.pop()
            used_mask = old_used
            
        return False

    # Multiple restarts with different random seeds to escape local minima
    # We use the deadline to limit restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        random.seed(restarts * 12345 + M)
        used_mask = 0
        rows.clear()
        
        # Shuffle the order of rows? No, rows are identical in structure.
        # But the *differences* they consume matter.
        
        if backtrack(0):
            return rows
            
    return None

def _generate_random_row(k, M, used_mask):
    """
    Generate a random row of k+1 increasing integers starting at 0, max <= M,
    such that its differences do not conflict with used_mask.
    """
    # We need to pick k values from 1..M.
    # To make it faster, we can prune early.
    
    # Simple approach: Sample k values, sort, check.
    # Optimization: If M is small, we can iterate? No, sampling is fine.
    
    for _ in range(50):  # Try 50 times to find a valid row
        # Sample k distinct values from 1..M
        if M < k:
            return None
            
        vals = random.sample(range(1, M + 1), k)
        vals.sort()
        row = [0] + vals
        
        # Check differences
        valid = True
        local_mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                bit = 1 << d
                if (used_mask & bit) or (local_mask & bit):
                    valid = False
                    break
                local_mask |= bit
            if not valid:
                break
        
        if valid:
            return row
            
    return None

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    This provides a fast upper bound.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False

        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
