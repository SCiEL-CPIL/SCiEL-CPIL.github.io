import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a local search (simulated annealing / hill climbing) approach to minimize
    the largest entry while ensuring all pairwise differences across rows are distinct.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
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

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # Helper: compute all pairwise differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: compute union of all diffs across all rows
    def all_diffs(rows):
        s = set()
        for r in rows:
            s |= row_diffs(r)
        return s

    # Helper: check if a candidate row is valid given the set of used diffs
    # (excluding its own current diffs, since we are replacing it)
    def is_row_valid(row, used_diffs_other_rows):
        rd = row_diffs(row)
        if len(rd) != len(row) * (len(row) - 1) // 2:
            return False  # internal duplicates
        if rd & used_diffs_other_rows:
            return False  # conflict with other rows
        return True

    # Helper: compute scope (max entry)
    def scope(rows):
        return max(max(r) for r in rows)

    # If not the specific hard case, just return greedy
    if n_rows != 7 or k != 5:
        return best_solution

    # Specific solver for (7, 5)
    # We will use a local search strategy:
    # 1. Start from a valid configuration (greedy).
    # 2. Try to reduce the scope by modifying rows.
    # 3. If stuck, try a random restart or simulated annealing move.

    # We aim for a scope <= 112 (best known) or ideally 105 (lower bound).
    # Let's try to find a solution with scope 105 first. If not, 106, 107, etc.
    
    # A better approach for DTS(7,5):
    # The number of differences per row is 15. Total 105 differences.
    # If scope is M, the available differences are 1..M.
    # For M=105, we must use ALL differences 1..105 exactly once.
    # This is a very tight constraint.
    
    # Let's implement a randomized local search that tries to minimize the max entry.
    # State: 7 rows of 6 integers.
    # Move: change one row to a new valid row with a smaller max entry? 
    # Or change one element in a row? Changing one element breaks the increasing order and diff constraints.
    # It's better to regenerate entire rows.
    
    # Strategy:
    # 1. Generate many candidate rows with small max entries.
    # 2. Use a matching / backtracking to select 7 non-conflicting rows with minimal total/max scope.
    
    # Let's try a beam search or iterative deepening on the max entry M.
    # For a fixed M, find 7 rows with max <= M and disjoint diffs.
    
    def solve_with_scope(M, time_limit):
        """
        Finds a DTS(7,5) with max entry <= M.
        Returns list of rows or None.
        """
        # Pre-generate candidate rows? 
        # Number of possible rows of length 6 from 0..M is C(M, 5) which is large for M=105.
        # C(105,5) ~ 96 million. Too many to pre-generate all.
        
        # We will use a backtracking search with good pruning.
        # Sort rows by their max entry to prioritize smaller scopes?
        
        # To make this efficient, we can use a randomized restart approach with a time limit.
        # But for M=105, the solution space is extremely constrained.
        
        # Let's use a simple randomized local search that starts from a greedy solution 
        # and tries to reduce the scope.
        
        # Actually, for M=105, a dedicated solver might be better.
        # But since we have time, let's try a multi-start local search.
        
        current_rows = greedy_construct(7, 5)
        current_scope = scope(current_rows)
        
        if current_scope <= M:
            return current_rows
        
        # Simulated Annealing
        # State: rows
        # Cost: scope (max entry) + penalty for invalidity (though we always keep valid)
        # Moves: 
        #   - Regenerate one row with a constraint that its max entry is lower?
        #   - Or just regenerate a row randomly within [0, M] and check validity.
        
        # Let's try to replace rows one by one with "better" rows (lower max entry).
        
        def find_better_row(used_diffs, current_row, M):
            """
            Tries to find a row with max entry < current_row[-1] 
            (or just low) that is valid given used_diffs.
            Returns None if not found quickly.
            """
            # Random search for a valid row
            # We want a row of 6 ints: 0 = a0 < a1 < ... < a5 <= M
            # such that all 15 diffs are in [1, M] and disjoint from used_diffs.
            
            # Heuristic: try rows with small max entry.
            # We can sample random subsets of size 5 from {1..M} and add 0.
            
            # But we want to be faster.
            # Let's try to find ANY valid row with max <= M.
            # If we find one, we replace the current row.
            # To minimize scope, we should prefer rows with smaller max.
            
            # For now, let's just find a valid row.
            
            # Randomized search
            for _ in range(500):
                # Sample 5 distinct numbers from 1..M
                # To bias towards smaller numbers, we can sample from 1..M//2 sometimes?
                # Or just uniform.
                # Actually, if M is small, uniform is fine.
                
                candidates = random.sample(range(1, M + 1), 5)
                candidates.sort()
                new_row = [0] + candidates
                
                # Check validity
                if is_row_valid(new_row, used_diffs):
                    return new_row
            return None
        
        # We will iterate, trying to reduce the max scope.
        # We track the best solution found.
        
        best_found = None
        best_scope_found = current_scope
        
        while time.time() < time_limit:
            # Identify the row with the largest max entry
            max_idx = max(range(7), key=lambda i: current_rows[i][-1])
            current_max = current_rows[max_idx][-1]
            
            # Compute used diffs excluding this row
            used_diffs = all_diffs(current_rows)
            current_diffs = row_diffs(current_rows[max_idx])
            used_diffs -= current_diffs
            
            # Try to find a better row
            # We want a row with max entry < current_max (if possible)
            # Or just any row that leads to a lower overall scope.
            
            # Let's try to find a row with max entry in [1, current_max - 1]
            # If found, replace.
            
            improved = False
            for _ in range(20): # 20 attempts per step
                # Target max entry range
                # We want to lower the scope.
                # Let's try random rows with max < current_max
                if current_max > 1:
                    upper_bound = current_max - 1
                else:
                    upper_bound = M
                
                if upper_bound < 5: # Need at least 1..5
                    break
                
                candidates = random.sample(range(1, upper_bound + 1), 5)
                candidates.sort()
                new_row = [0] + candidates
                
                if is_row_valid(new_row, used_diffs):
                    # Check if this actually improves the global scope
                    # The new global scope is max(max(rows except i), new_row[-1])
                    other_max = max(max(r[-1]) for i, r in enumerate(current_rows) if i != max_idx)
                    new_scope = max(other_max, new_row[-1])
                    
                    # If new_scope < current_scope, it's an improvement
                    # Even if not strictly smaller, it might help future moves.
                    # Let's accept if new_row[-1] < current_max
                    if new_row[-1] < current_max:
                        current_rows[max_idx] = new_row
                        improved = True
                        # Update best
                        current_scope = scope(current_rows)
                        if current_scope < best_scope_found:
                            best_scope_found = current_scope
                            best_found = [r[:] for r in current_rows]
                        break # Try next row
            
            if not improved:
                # Stuck. Try a random restart or perturbation.
                # Perturbation: randomly regenerate one row (even if it increases scope)
                # This helps escape local minima.
                if random.random() < 0.1:
                    idx = random.randint(0, 6)
                    used_diffs = all_diffs(current_rows)
                    current_diffs = row_diffs(current_rows[idx])
                    used_diffs -= current_diffs
                    
                    # Find any valid row
                    for _ in range(100):
                        candidates = random.sample(range(1, M + 1), 5)
                        candidates.sort()
                        new_row = [0] + candidates
                        if is_row_valid(new_row, used_diffs):
                            current_rows[idx] = new_row
                            break
        
        return best_found if best_found else (current_rows if scope(current_rows) <= M else None)

    # Iterate M from 105 upwards
    for M in range(105, 130):
        if time.time() > deadline - 5:
            break
            
        # Give each M a time slice.
        # If M=105 fails quickly, move to 106.
        # Let's allocate 10s per M, but cap at deadline.
        time_limit = min(deadline - 5, start_time + (M - 104) * 5) # Increasing time?
        # No, just fixed slices.
        time_limit = start_time + 10
        
        sol = solve_with_scope(M, time_limit)
        
        if sol:
            # Verify
            all_diffs = set()
            valid = True
            for row in sol:
                if len(row) != 6 or row[0] != 0:
                    valid = False
                    break
                if not all(row[i] < row[i+1] for i in range(5)):
                    valid = False
                    break
                for i in range(6):
                    for j in range(i+1, 6):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
                if max(row) > M:
                    valid = False
                    break
            
            if valid:
                return sol

    return best_solution

# EVOLVE-BLOCK-END
