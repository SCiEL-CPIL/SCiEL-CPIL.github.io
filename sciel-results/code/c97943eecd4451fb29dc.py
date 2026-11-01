import time
import random
import math
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145.0
    
    # Total number of differences needed
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope M
    M_lower = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    # We will search for the smallest M starting from M_lower
    # The CSP approach is exact, so if it finds a solution for M, it's optimal for that M.
    # We iterate M increasing until we find a valid configuration.
    
    best_sol = None
    best_M = None
    
    # Strategy:
    # 1. For a given M, generate all valid "difference sets" (rows) that fit in [0, M].
    #    A row is a set of k+1 integers {0, x1, ..., xk} with 0 < x1 < ... < xk <= M.
    #    The difference set D(row) = {xj - xi | 0 <= i < j <= k}.
    #    Validity condition: D(row1) and D(row2) must be disjoint for all rows.
    #    This is equivalent to finding n_rows rows with pairwise disjoint difference sets.
    #    
    # 2. Instead of generating all rows (which is C(M, k) - huge), we use backtracking.
    #    We build rows one by one.
    #    State: used_diffs (set of integers).
    #    Goal: Construct n_rows rows.
    #    
    #    To make this efficient, we need a good way to generate candidate rows 
    #    that avoid used_diffs.
    #    
    #    Heuristic: 
    #    - Start with used_diffs = empty.
    #    - For row 1, pick a random valid row.
    #    - For row 2, pick a random valid row that doesn't conflict with row 1.
    #    - ...
    #    - If stuck, backtrack.
    #    
    #    However, pure backtracking on "pick a row" is slow because the branching factor is huge.
    #    
    #    Alternative CSP formulation:
    #    Variables: The elements of the rows.
    #    But rows are not independent.
    #    
    #    Let's try a hybrid:
    #    1. Pre-compute a list of "good" candidate rows? No, M is variable.
    #    2. Use Randomized Greedy Construction with Backtracking on "stuck" points.
    
    def solve_csp(M):
        """
        Attempts to find n_rows disjoint difference sets within scope M.
        Uses randomized backtracking with MRV-like heuristics for row construction.
        """
        # Time check inside solver
        if time.time() > deadline - 2:
            return None
        
        # We represent the state by the set of used differences.
        # To allow backtracking, we need to undo.
        # Since we are building rows sequentially, we can keep a stack of rows.
        
        # Optimization: 
        # Instead of picking a whole row at once, we can pick elements?
        # No, picking a whole row is a good atomic move if we can generate them efficiently.
        
        # How to generate a row that avoids a specific set of differences?
        def generate_row_avoiding(used_diffs, M, k):
            """
            Generates a random row [0, x1, ..., xk] such that all differences 
            are not in used_diffs and are distinct within the row.
            """
            # Random walk with backtracking within the row generation?
            # Or just try random subsets?
            
            # Let's try a greedy random walk with local backtracking.
            # This is essentially a CSP solver for a single row.
            
            row = [0]
            # We need to pick k more numbers from 1..M.
            # To make it faster, we can pre-filter candidates?
            
            # Let's use a simple recursive backtracking for the row itself.
            # Depth k.
            # At each step, pick a next value > last.
            
            def backtrack_row(current_row, last_val):
                if len(current_row) == k + 1:
                    return current_row[:]
                
                # MRV heuristic for the row? 
                # The "domain" for the next value is (last_val, M].
                # But we must ensure all diffs are valid.
                
                # To speed up, we can randomize the order of candidates.
                # Range: last_val + 1 to M - (k - len(current_row) + 1)
                # Min next: last_val + 1
                # Max next: M - (k - len(current_row))  # leave 1 for each remaining
                
                min_next = last_val + 1
                max_next = M - (k - len(current_row))
                
                if min_next > max_next:
                    return None
                
                # Generate candidates in random order
                # To avoid generating too many, we can limit attempts or use a smarter step.
                # For small k (like 7), the space is manageable.
                
                candidates = list(range(min_next, max_next + 1))
                random.shuffle(candidates)
                
                for cand in candidates:
                    # Check conflicts with current_row
                    valid = True
                    for a in current_row:
                        d = cand - a
                        if d in used_diffs or d in set(current_row[i] - current_row[j] for i in range(len(current_row)) for j in range(i+1, len(current_row))):
                            # Note: internal diffs are always distinct if we check strictly? 
                            # No, we must check against used_diffs (global) AND internal consistency.
                            # Actually, if we build strictly increasing, internal diffs are distinct? 
                            # No, as established before, they can repeat. 
                            # So we must check if d is already in the set of internal diffs of current_row.
                            # To be efficient, maintain a set of internal diffs.
                            valid = False
                            break
                            
                    if valid:
                        # Also check against global used_diffs
                        # The loop above checks `d in used_diffs`.
                        # We also need to ensure `d` is not in the internal diffs of `current_row`.
                        # Let's refactor to pass internal_diffs set.
                        pass
                        
                return None
            
            # The above is a bit messy. Let's do a cleaner recursive row builder.
            
            def build_row(idx, last_val, current_row, internal_diffs):
                if idx == k:
                    return current_row[:]
                
                min_next = last_val + 1
                max_next = M - (k - idx)
                
                if min_next > max_next:
                    return None
                
                # Randomize candidates
                # To avoid iterating 1..M every time, we can sample.
                # But for correctness in backtracking, we should eventually try all.
                # Given k=7, M~130, the branching factor is large but depth is small.
                
                # Heuristic: Try values that are "far" from existing to reduce density?
                # Or just random shuffle.
                
                candidates = list(range(min_next, max_next + 1))
                random.shuffle(candidates)
                
                for cand in candidates:
                    # Check global conflicts
                    conflict_global = False
                    for a in current_row:
                        d = cand - a
                        if d in used_diffs:
                            conflict_global = True
                            break
                    if conflict_global:
                        continue
                        
                    # Check internal conflicts
                    # New internal diffs are cand - a for a in current_row
                    # Are any of these in internal_diffs?
                    conflict_internal = False
                    new_diffs = []
                    for a in current_row:
                        d = cand - a
                        if d in internal_diffs:
                            conflict_internal = True
                            break
                        new_diffs.append(d)
                            
                    if conflict_internal:
                        continue
                        
                    # Accept candidate
                    current_row.append(cand)
                    internal_diffs.update(new_diffs)
                    
                    res = build_row(idx + 1, cand, current_row, internal_diffs)
                    if res:
                        return res
                        
                    # Backtrack
                    current_row.pop()
                    internal_diffs.difference_update(new_diffs)
                    
                return None
                
            # Start building row
            # Initial state: row=[0], internal_diffs=set()
            return build_row(0, 0, [0], set())

        # Main Backtracking Loop for n_rows
        # We use a stack of (row, used_diffs_snapshot)
        # To save memory/time, we can just keep the current used_diffs and the rows list.
        
        rows = []
        used_diffs = set()
        
        # Recursive function to build the set of rows
        def build_rows(row_idx):
            if time.time() > deadline - 2:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Generate a candidate row
            # To diversify, we can try multiple random rows if the first fails?
            # The build_row function is already randomized.
            
            # However, build_row might return None quickly if it gets stuck.
            # We should retry with different random seeds/shuffles.
            
            # Limit retries for this specific row_idx to avoid infinite loops on bad M
            max_retries = 50
            for _ in range(max_retries):
                row = generate_row_avoiding(used_diffs, M, k)
                if row is None:
                    continue
                
                # Check if this row is valid (should be by construction)
                # Add to used_diffs
                new_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        new_diffs.add(row[j] - row[i])
                
                # Check if any conflict (should be none if generate_row_avoiding worked correctly)
                # But generate_row_avoiding only checks against `used_diffs` passed in.
                # So it should be disjoint.
                
                rows.append(row)
                used_diffs.update(new_diffs)
                
                if build_rows(row_idx + 1):
                    return True
                
                # Backtrack
                rows.pop()
                used_diffs.difference_update(new_diffs)
                
            return False
            
        if build_rows(0):
            return rows
        return None

    # Try increasing M
    # We expect the answer to be near M_lower.
    # The best known for (4,7) is 134.
    # M_lower for n=4, k=7: total_diffs = 4 * 28 = 112.
    # M_lower = ceil((sqrt(1 + 8*112) - 1)/2) = ceil((sqrt(897)-1)/2) = ceil((29.9-1)/2) = ceil(14.4) = 15.
    # Wait, the counting bound 112 refers to the number of diffs, not M.
    # The problem statement says "simple counting bound is 112".
    # This usually means M must be at least something?
    # If total diffs is 112, and they are distinct integers from 1..M, then M >= 112.
    # So M_lower should be 112.
    
    # Let's re-calculate M_lower based on the distinct differences requirement.
    # The differences are distinct positive integers.
    # The smallest possible set of differences is {1, 2, ..., total_diffs}.
    # The maximum difference in any row is at most M.
    # So we must have M >= total_diffs? No.
    # The differences are a subset of {1, ..., M}.
    # So we need M >= total_diffs.
    # For n=4, k=7, total_diffs = 112. So M >= 112.
    
    # The "simple counting bound is 112" likely refers to M.
    # Best known is 134.
    
    M_lower = total_diffs
    
    for offset in range(0, 50):
        M = M_lower + offset
        
        if time.time() > deadline - 5:
            break
            
        # Try to solve for this M
        # We can do a few restarts of the randomized backtracking
        sol = None
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            sol = solve_csp(M)
            if sol is not None:
                best_sol = sol
                best_M = M
                break
        
        if best_sol is not None:
            break
    
    if best_sol is None:
        # Fallback: Greedy construction with large M to ensure validity
        # This ensures we return *a* valid solution, even if not optimal.
        # Score will be low, but better than 0.
        M_fallback = total_diffs + 200
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Pick the smallest available x
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        # Also check internal uniqueness? 
                        # If we pick smallest x, and d is not in used_diffs, 
                        # is it possible d is in internal_diffs?
                        # Yes. So we must check internal diffs too.
                        internal_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                internal_diffs.add(row[j] - row[i])
                                
                        if any((x - a) in internal_diffs for a in row):
                            valid = False
                            
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
            rows.append(row)
        
        best_sol = rows
    
    return best_sol
