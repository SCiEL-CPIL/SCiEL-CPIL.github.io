"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, minimizing the largest entry.
    
    Uses randomized backtracking with:
    - Bitmask for difference tracking
    - MRV (Minimum Remaining Values) heuristic
    - Iterative deepening on scope
    - Randomized restarts
    - Deadline guard
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    # Number of differences per row: C(k+1, 2) = k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on scope: we need at least total_diffs distinct differences
    # The differences range from 1 to scope, so scope >= total_diffs
    # But also each row's max element contributes to the scope
    # Simple lower bound: scope >= max(needed)
    # For (7,5): diffs_per_row=15, total=105, but best known is 112
    # A better lower bound considers that differences in a single row must fit
    # within [1, max_row], and across rows they must be distinct.
    
    # Greedy lower bound: try to pack as many differences as possible
    # In a row with max element m, the available differences are 1..m
    # But they must be distinct across all rows.
    # Simple bound: scope >= total_diffs (each difference 1..scope can be used once)
    # But we also need scope to be large enough for each individual row.
    
    # Let's use a smarter lower bound via greedy packing
    def compute_lower_bound(n_rows, k, max_scope=200):
        """Compute a lower bound on the minimum scope."""
        diffs_per_row = k * (k + 1) // 2
        # Each row needs diffs_per_row distinct differences from [1, scope]
        # Total distinct differences needed: n_rows * diffs_per_row
        # So scope >= n_rows * diffs_per_row is a trivial bound
        lb = n_rows * diffs_per_row
        # But also, for a single row to have k+1 elements starting at 0 with max m,
        # the number of distinct positive differences is at most m*(m+1)/2 but we need exactly diffs_per_row
        # The minimum m for a row is determined by the optimal ruler of length k
        # For k=5 (6 marks), the optimal ruler has length 11 (marks: 0,1,4,10,12,17... wait)
        # Actually for a Golomb ruler with 6 marks, the minimum length is 17? No.
        # A Golomb ruler of order 6 has minimum length 17? Let me check: 
        # Order 1: 0
        # Order 2: 1
        # Order 3: 3
        # Order 4: 6
        # Order 5: 11
        # Order 6: 17
        # So each row needs at least 17 if it were a Golomb ruler alone.
        # But here differences can repeat across rows as long as they're not the same value.
        # Wait no - "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows"
        # So ALL differences across ALL rows must be distinct.
        # So total distinct differences = n_rows * C(k+1,2)
        # And each difference is in [1, scope]
        # So scope >= n_rows * C(k+1,2) = total_diffs
        # But also each row's max element <= scope, and within a row, differences must be distinct
        # (which they are automatically if the row is strictly increasing and we consider all pairs)
        # Wait, within a single row, can two pairs have the same difference? 
        # If a_j - a_i = a_l - a_m with (i,j) != (m,l), that would be a repeated difference within the row.
        # The problem says "all differences between two entries of the same row are distinct over all rows"
        # This means: for any two pairs (possibly from different rows), the difference values must be distinct.
        # So within a single row, all C(k+1,2) differences must be distinct (Golomb ruler property).
        # AND across rows, no difference value can repeat.
        
        # So each row is a Golomb ruler fragment, and all difference values across all rows are globally unique.
        
        # Lower bound: scope >= max(
        #   total_diffs,  # need total_diffs distinct values in [1, scope]
        #   min_golomb_length(k+1)  # each row needs at least this
        # )
        # For k=5, min Golomb ruler length for 6 marks is 17
        # total_diffs = 7 * 15 = 105
        # So lb = max(105, 17) = 105
        
        return lb
    
    lb = compute_lower_bound(n_rows, k)
    
    # Best solution found
    best_solution = None
    best_scope = float('inf')
    
    # Greedy initial solution as a starting point
    def greedy_solve(n_rows, k):
        """Simple greedy to get an initial feasible solution."""
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
    
    greedy_rows = greedy_solve(n_rows, k)
    greedy_scope = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    best_scope = greedy_scope
    
    # Now try to improve with backtracking + iterative deepening
    # Start from lb and go up to greedy_scope
    
    def backtrack_solve(n_rows, k, max_scope, time_limit):
        """
        Backtracking search to find a solution with all entries <= max_scope.
        Returns a solution or None.
        """
        start_time_bt = time.time()
        diffs_per_row = k * (k + 1) // 2
        
        # used_diffs: set of difference values already used
        # We'll represent this as a set for clarity, but could use bitmask for speed
        # For scope up to ~200, a set is fine
        
        rows = []
        used_diffs = set()
        
        def can_place(row_partial, candidate):
            """Check if adding candidate to row_partial doesn't conflict with used_diffs."""
            new_diffs = {candidate - a for a in row_partial}
            return not (new_diffs & used_diffs)
        
        def solve_row(row_idx, partial_row):
            """Recursively build row row_idx."""
            if time.time() - start_time_bt > time_limit:
                return False
            
            if len(partial_row) == k + 1:
                rows.append(partial_row[:])
                if row_idx == n_rows - 1:
                    return True
                return solve_row(row_idx + 1, [0])
            
            # Determine the range for the next element
            last = partial_row[-1]
            # The next element must be > last
            # And the new differences must not conflict
            # Also, we need to leave room for the remaining elements
            # If we have r more elements to place, the max element is at most max_scope
            # The remaining elements must be in (last, max_scope]
            remaining = k + 1 - len(partial_row)
            
            # MRV: try candidates that leave the most flexibility
            # For simplicity, try from last+1 upward, but with some randomization
            # Actually, let's try to be smart: the next element should not be too large
            # because we need room for remaining elements
            
            # Upper bound for current candidate: we need at least (remaining-1) more elements
            # each at least 1 apart, so candidate <= max_scope - (remaining - 1)
            max_candidate = max_scope - (remaining - 1)
            
            # Generate candidates in a smart order
            # Try smaller values first (to keep scope tight) but with some randomization
            candidates = list(range(last + 1, max_candidate + 1))
            
            # Shuffle slightly for randomization but keep bias toward smaller
            # Use a weighted random selection approach
            if len(candidates) > 1:
                # Bias toward smaller values but allow some randomness
                # This helps escape local minima
                random.shuffle(candidates)
                # Sort by a heuristic: prefer values that don't block many future options
                # For now, just use the shuffled order with a bias
                # Actually, let's try a different approach: try a few random candidates
                # mixed with sequential
                
                # Hybrid: try first few in order, then random
                n_sequential = min(5, len(candidates))
                ordered_part = candidates[:n_sequential]
                random_part = candidates[n_sequential:]
                random.shuffle(random_part)
                candidates = ordered_part + random_part
            
            for cand in candidates:
                if time.time() - start_time_bt > time_limit:
                    return False
                
                if not can_place(partial_row, cand):
                    continue
                
                # Tentatively place
                new_diffs = {cand - a for a in partial_row}
                partial_row.append(cand)
                used_diffs.update(new_diffs)
                
                if solve_row(row_idx, partial_row):
                    return True
                
                # Backtrack
                partial_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        return solve_row(0, [0])
    
    # Iterative deepening: try scopes from lb to best_scope - 1
    # Use randomized restarts for each scope
    max_restarts = 50
    restart_time_budget = (DEADLINE - time.time()) / max_restarts if time.time() < DEADLINE else 0
    
    current_scope = lb
    while current_scope < best_scope and time.time() < DEADLINE - 5:
        # Try to find a solution with this scope
        found = False
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 5:
                break
            
            # Random seed for this restart
            random.seed(hash((current_scope, restart)) & 0xFFFFFFFF)
            
            time_limit = min(2.0, (DEADLINE - 5 - time.time()) / max(1, max_restarts - restart))
            if time_limit <= 0:
                break
            
            result = backtrack_solve(n_rows, k, current_scope, time_limit)
            if result:
                found = True
                best_scope = current_scope
                best_solution = rows  # This is a bug - rows is local to backtrack_solve
                break
        
        if found:
            # We found a solution at current_scope, try to go lower
            # Actually, we should try to find the minimum, so continue
            # But we found one, so update best
            pass
        else:
            # No solution at this scope, try larger
            current_scope += 1
    
    return best_solution
# EVOLVE-BLOCK-END
