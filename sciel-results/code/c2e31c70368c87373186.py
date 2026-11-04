"""
Strategy: Constraint Satisfaction Problem (CSP) with Backtracking and Heuristic Ordering.
Instead of generating random rows and fixing conflicts, we construct the solution incrementally.
We iterate over the maximum scope M from the theoretical lower bound upwards.
For a fixed M, we use a backtracking search to find n_rows disjoint sets of differences.
The search builds rows one by one. Within a row, it picks the next element by trying 
candidates in an order that minimizes the "degree" of future constraint violations 
(i.e., preferring candidates that use "rare" differences or leave more options for others).
"""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Theoretical lower bound for scope M
    # We need total_diffs_needed distinct differences from 1 to M.
    # So M >= total_diffs_needed.
    # Also, the largest element in a row of size k+1 starting at 0 is at least k (if 0,1,2...).
    # More tightly, if we have n_rows rows, the total number of differences is n_rows * C(k+1, 2).
    # These differences must be distinct integers in [1, M].
    # So M >= n_rows * C(k+1, 2).
    
    lower_bound = total_diffs_needed
    
    # For (7, 5), total_diffs_needed = 7 * 15 = 105.
    # Best known is 112.
    
    def get_row_diffs(row):
        """Return list of all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences are distinct."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_row_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    def compute_scope(rows):
        return max(max(row) for row in rows)

    # Precompute all valid rows for a given scope M?
    # No, that's C(M, k) which is large.
    # Instead, we generate rows on the fly during backtracking.
    
    # To make backtracking efficient, we need to quickly check if a partial row
    # can be extended to a full row without conflicting with used differences.
    
    # Let's define a solver for a fixed M.
    def solve_for_scope(M, seed):
        rng = random.Random(seed)
        
        # We need to select n_rows disjoint sets of differences.
        # Each set corresponds to a row of length k+1 starting at 0.
        
        # We will build rows one by one.
        # State:
        # used_diffs: set of differences already used by previous rows.
        # current_row: list of elements in the current row being built.
        # row_index: which row we are currently building (0 to n_rows-1).
        
        # Heuristic: 
        # When building a row, we pick the next element x > current_row[-1].
        # The new differences are x - y for y in current_row.
        # These must not be in used_diffs.
        
        # To guide the search, we can try to build rows that are "sparse" or "dense"?
        # Actually, a common heuristic for difference sets is to try to keep the
        # differences "balanced" or to use a specific structure.
        
        # Let's use a simple backtracking with randomization for tie-breaking.
        
        def backtrack(row_idx, used_diffs_mask, current_rows):
            if time.time() > DEADLINE - 5:
                return None
            
            if row_idx == n_rows:
                return current_rows[:]
            
            # Build the row for row_idx
            # Start with [0]
            row = [0]
            
            # We need to add k more elements.
            # Let's use a recursive helper to build the row.
            
            def build_row_partial(row, depth):
                if time.time() > DEADLINE - 5:
                    return None
                
                if depth == num_elements:
                    return row[:]
                
                last_val = row[-1]
                
                # Candidate next values: last_val + 1 to M
                # We need to check if the differences created are valid.
                # To improve efficiency, we can sort candidates by some heuristic.
                # Heuristic: Prefer candidates that use smaller differences? 
                # Or prefer candidates that don't conflict with "critical" differences?
                
                # For now, let's try values in increasing order, but with some randomization
                # to avoid getting stuck in a deterministic local minimum.
                
                candidates = list(range(last_val + 1, M + 1))
                
                # Apply a heuristic sort:
                # Calculate the "cost" of adding x.
                # Cost = number of new differences that are "small" (more likely to conflict with others)?
                # Or just randomize?
                
                # Let's try a mix: 50% random, 50% increasing.
                if rng.random() < 0.5:
                    rng.shuffle(candidates)
                
                for x in candidates:
                    # Check if x is valid
                    valid = True
                    new_diffs = []
                    for y in row:
                        d = x - y
                        if d in used_diffs_mask:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Pruning: Check if it's possible to complete the row.
                    # We need num_elements - depth - 1 more elements.
                    # The minimum additional differences required is... complicated.
                    # Simple pruning: If the number of remaining available differences
                    # in [1, M] is less than what we need for the rest of the rows, fail.
                    # But this is global, not local to the row.
                    
                    # Local pruning: 
                    # The next elements must be > x.
                    # The smallest possible next difference is 1.
                    # If we have used all differences in [1, M-x] that are needed?
                    # Let's skip complex pruning for now and rely on backtracking.
                    
                    # Try this x
                    new_used_mask = used_diffs_mask | set(new_diffs)
                    result = build_row_partial(row + [x], depth + 1)
                    if result is not None:
                        return result
            
            # Start building the row
            new_row = build_row_partial(row, 1)
            
            if new_row is None:
                return None
            
            # Update state
            row_diffs = get_row_diffs(new_row)
            new_used_mask = used_diffs_mask | set(row_diffs)
            
            # Recurse to next row
            result = backtrack(row_idx + 1, new_used_mask, current_rows + [new_row])
            if result is not None:
                return result
            
            return None

        # Initial call
        initial_rows = []
        return backtrack(0, set(), initial_rows)

    # Main loop: Iterate M from lower_bound upwards
    best_rows = None
    best_scope = float('inf')
    
    # We can also use a randomized restart strategy within each M
    # Or just try M, M+1, M+2...
    
    # To be robust, let's try a few seeds for each M if the first fails.
    
    current_M = lower_bound
    max_M = 200 # Safety upper bound
    
    while current_M <= max_M:
        if time.time() > DEADLINE - 10:
            break
            
        # Try to find a solution for this M
        # Use a few random seeds to diversify the search
        found = False
        for seed in range(5):
            if time.time() > DEADLINE - 5:
                break
            rows = solve_for_scope(current_M, seed)
            if rows is not None:
                # Verify validity
                if is_valid(rows):
                    best_rows = rows
                    best_scope = current_M
                    found = True
                    break
        
        if found:
            # We found a solution for this M.
            # Since we are iterating M upwards, this is the minimal M for this search path.
            # However, due to randomization, we might have missed a solution for a smaller M
            # if the search space was not fully explored. 
            # But backtracking is complete for a fixed M if it runs to completion.
            # The only reason to fail is if no solution exists for that M.
            # So if we find a solution, it is optimal for this run.
            return best_rows
        
        current_M += 1
        
    # If we didn't find a solution in the loop, return the best we have or a fallback
    if best_rows is not None:
        return best_rows
    
    # Fallback: Generate a valid solution with large scope
    # This ensures we always return a valid result
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

# EVOLVE-BLOCK-END
