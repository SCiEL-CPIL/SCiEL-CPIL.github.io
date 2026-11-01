import time
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    
    Strategy: 
    1. Estimate lower bound M.
    2. Iterate M upwards.
    3. For each M, generate all valid "Sidon rows" (subsets of size k+1 from 0..M 
       containing 0, with distinct internal differences).
    4. Represent each row's differences as a bitmask.
    5. Use backtracking search to find n_rows rows with disjoint difference masks.
    """
    deadline = time.time() + 145.0
    
    # Lower bound: total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Simple counting bound: M(M+1)/2 >= total_diffs_needed is not quite right because 
    # differences don't have to be 1..M, just distinct. 
    # But max difference is M. So we need total_diffs_needed distinct values in {1..M}.
    # So M >= total_diffs_needed.
    M_start = max(total_diffs_needed, k * (k + 1) // 2 + k, 50)
    
    # Upper bound from a simple greedy construction
    greedy_rows = _greedy_construct(n_rows, k)
    M_end = _max_entry(greedy_rows)
    
    best_rows = greedy_rows
    best_M = M_end
    
    # If greedy is already very good, we might not need to search much.
    # But we want to find the minimum.
    
    for M in range(M_start, M_end):
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution for this M
        rows = _solve_exact_packing(n_rows, k, M, deadline)
        
        if rows is not None:
            best_rows = rows
            best_M = M
            # Since we iterate M upwards, the first solution found is optimal for this M.
            # Is it globally optimal? Yes, if we checked all smaller M.
            # So we can return immediately.
            return rows
            
    return best_rows

def _solve_exact_packing(n_rows, k, M, deadline):
    """
    Find n_rows disjoint Sidon rows from {0..M}.
    """
    # 1. Generate all valid rows
    # A row is a tuple (val1, ..., valk) where 0 < val1 < ... < valk <= M
    # Full row is (0, val1, ..., valk)
    # Internal differences must be distinct.
    
    candidates = []
    
    # Optimization: Generate combinations of k values from 1..M
    # Check if they form a Sidon set with 0.
    # Differences are:
    # v_i - 0 = v_i
    # v_j - v_i (j>i)
    # All these must be distinct.
    
    # To speed up generation, we can use a heuristic or just brute force if M is small.
    # For M=112, k=5, C(112,5) is ~2.5M. This might be slow in Python loop.
    # Let's try to optimize the check.
    
    # We will generate candidates and store (mask, row_tuple)
    # mask is a Python int representing the set of differences.
    
    # To avoid memory explosion, we might limit the number of candidates 
    # or use a smarter generation.
    # However, for k=5, the number of valid Sidon rows is significantly less than total combos.
    
    # Let's use a generator to avoid storing all combos at once, 
    # but we need to sort/filter them for the backtracking.
    # Actually, we can just collect them into a list.
    
    start_time = time.time()
    
    # We can prune generation: 
    # If we pick v1, v2, ..., vk
    # The differences are v1, v2, ..., vk, v2-v1, v3-v1, ..., vk-v1, v3-v2, ..., vk-v(k-1)
    
    # Let's iterate.
    # To make it faster, we can break early in the check.
    
    for combo in combinations(range(1, M + 1), k):
        if time.time() > deadline - 5:
            return None # Timeout in generation, skip this M
        
        # Check Sidon property
        diffs = set()
        valid = True
        # Add differences with 0
        for v in combo:
            if v in diffs:
                valid = False
                break
            diffs.add(v)
        
        if not valid:
            continue
            
        # Add differences between combo elements
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                d = combo[j] - combo[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
                
        if valid:
            # Convert diffs to bitmask
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            candidates.append((mask, (0,) + combo))
            
    if len(candidates) < n_rows:
        return None
        
    # 2. Backtracking search
    # Sort candidates to improve pruning?
    # Heuristic: Rows with "smaller" max value might be better? 
    # Or rows with masks that have bits set in "lower" positions?
    # Let's sort by the max value in the row (last element).
    candidates.sort(key=lambda x: x[1][-1])
    
    # Further optimization: 
    # If two rows have the same mask, we only need one? 
    # No, the rows themselves are different, but if they have the same diff set, 
    # they are interchangeable in terms of constraints. 
    # We can deduplicate by mask to reduce search space?
    # If we have two rows with same mask, picking one is equivalent to picking the other 
    # for the purpose of satisfying the "disjoint differences" constraint.
    # However, we need distinct rows? The problem says "Choose 7 rows". 
    # It doesn't explicitly say rows must be distinct from each other, 
    # but "all differences ... are distinct over all rows" implies 
    # if row A and row B are identical, their differences are identical, 
    # so they would conflict unless the difference set is empty (impossible).
    # So rows MUST be disjoint in differences. 
    # Thus, no two rows can have the same difference set.
    # So we can deduplicate candidates by mask.
    
    unique_masks = {}
    for mask, row in candidates:
        if mask not in unique_masks:
            unique_masks[mask] = row
            
    unique_candidates = [(mask, row) for mask, row in unique_masks.items()]
    
    if len(unique_candidates) < n_rows:
        return None
        
    # Sort unique candidates
    unique_candidates.sort(key=lambda x: x[1][-1])
    
    masks = [c[0] for c in unique_candidates]
    rows = [c[1] for c in unique_candidates]
    n_cands = len(unique_candidates)
    
    # DFS
    # We need to pick n_rows indices from 0..n_cands-1 such that masks are disjoint.
    # To avoid symmetry, we enforce index ordering: i1 < i2 < ... < in
    
    selected_indices = []
    
    def dfs(start_idx, count, used_mask):
        if time.time() > deadline - 2:
            return False
            
        if count == 0:
            return True
            
        # Pruning: not enough candidates left
        if n_cands - start_idx < count:
            return False
            
        for i in range(start_idx, n_cands):
            m = masks[i]
            if used_mask & m:
                continue
                
            # Accept
            selected_indices.append(i)
            if dfs(i + 1, count - 1, used_mask | m):
                return True
            selected_indices.pop()
            
        return False
        
    if dfs(0, n_rows, 0):
        result = [rows[i] for i in selected_indices]
        return result
        
    return None

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
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
