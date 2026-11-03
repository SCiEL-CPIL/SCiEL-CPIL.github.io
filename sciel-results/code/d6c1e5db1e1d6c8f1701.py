import time
import random
import sys
import math
from collections import defaultdict

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_construct(n, k_val, rng):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            x = row[-1]
            for pos in range(k_val):
                while True:
                    x += 1
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        break
            rows.append(row)
        return rows

    def csp_backtrack(n_rows, k_val, max_scope, rng):
        """
        Formulate as CSP: assign each of the total_diffs slots to a unique d in [1, max_scope].
        Slots are grouped by row. For each row, the set of assigned diffs must form a valid difference set.
        We use MRV to select the next slot to assign.
        """
        # Precompute all possible difference pairs (i, j) for a row of size k+1
        # There are k*(k+1)/2 such pairs.
        # We will assign a value to each pair.
        # The constraint is that for each row, the set of values must be realizable by some increasing sequence starting at 0.
        
        # Actually, a simpler CSP formulation:
        # Variables: The elements of each row (excluding the first 0).
        # Domains: Increasing integers.
        # Constraints: Differences are unique globally.
        
        # Let's use a different CSP formulation which is often more effective for these problems:
        # Variables: The rows themselves? No, too complex.
        # Variables: The differences?
        
        # Let's try a backtracking search on the elements directly, but guided by MRV on the differences.
        # State:
        #   rows: list of lists (partially filled)
        #   used_diffs: set of all differences used so far
        
        # To use MRV effectively, we need to know which differences are "tight".
        # However, a standard backtracking on elements with forward checking is robust.
        
        # Let's implement a backtracking search that builds rows one by one.
        # Within a row, we pick elements sequentially.
        # To improve, we can use a heuristic: pick the row with the fewest valid completions first?
        # Or pick the next element in the current row that has the fewest valid candidates?
        
        # Given the time limit, a randomized restart backtracking with a hard scope bound is a good strategy.
        
        rows = []
        used_diffs = set()
        
        # Heuristic: Build rows in order of increasing "difficulty" or just sequentially?
        # Sequential is standard. Let's try to be smart about the next element in the current row.
        
        def backtrack(row_idx, pos_in_row, current_row):
            if time.time() > deadline - 2:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_in_row == k_val:
                # Row complete
                rows.append(current_row)
                return backtrack(row_idx + 1, 0, [0])
            
            # Determine candidates for current_row[pos_in_row]
            # It must be > current_row[pos_in_row - 1]
            # And max_scope bound
            # And differences with previous elements must not be in used_diffs
            
            last_val = current_row[pos_in_row - 1]
            min_val = last_val + 1
            max_val = max_scope
            
            # Generate candidates
            # To use MRV-like heuristic, we might want to try candidates that leave the most options for future steps?
            # For now, let's just iterate. To add randomness, we can shuffle candidates.
            
            candidates = list(range(min_val, max_val + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                # Check differences
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = val - a
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Forward check: Can we still complete this row?
                    # Simple check: Is there enough "room" for remaining k_val - pos_in_row elements?
                    # The maximum difference we can form is val - 0 = val.
                    # We need k_val - pos_in_row more elements.
                    # The smallest possible next element is val + 1.
                    # The largest possible difference will be at least val + (k_val - pos_in_row) - 0? No.
                    # Let's just do a quick heuristic check:
                    # The number of unused differences in [1, max_scope] must be >= remaining slots for this row + all future rows.
                    # This is expensive to check globally.
                    
                    # Just try the move
                    current_row.append(val)
                    used_diffs.update(new_diffs)
                    
                    if backtrack(row_idx, pos_in_row + 1, current_row):
                        return True
                    
                    # Undo
                    used_diffs.difference_update(new_diffs)
                    current_row.pop()
            
            return False
        
        # Initialize
        rows = []
        used_diffs = set()
        # Note: The nested function uses nonlocal variables, but in Python, we need to be careful with scope.
        # Let's restructure to avoid closure issues or use a class/state dict.
        
        state = {
            'rows': [],
            'used_diffs': set()
        }
        
        def bt(row_idx, pos_in_row, current_row):
            if time.time() > deadline - 2:
                return False
            
            if row_idx == n_rows:
                return True
            
            if pos_in_row == k_val:
                state['rows'].append(current_row)
                return bt(row_idx + 1, 0, [0])
            
            last_val = current_row[pos_in_row - 1]
            min_val = last_val + 1
            max_val = max_scope
            
            candidates = list(range(min_val, max_val + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = val - a
                    if d in state['used_diffs'] or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    current_row.append(val)
                    state['used_diffs'].update(new_diffs)
                    
                    if bt(row_idx, pos_in_row + 1, current_row):
                        return True
                    
                    state['used_diffs'].difference_update(new_diffs)
                    current_row.pop()
            
            return False
        
        if bt(0, 0, [0]):
            return state['rows']
        return None

    # Main logic
    rng = random.Random(42)
    
    # 1. Get an initial solution to establish an upper bound
    init_rows = greedy_construct(n_rows, k, rng)
    if not init_rows:
        return [[0] * (k + 1) for _ in range(n_rows)] # Fallback, likely invalid but prevents crash
    
    current_best_scope = max(max(r) for r in init_rows)
    best_rows = init_rows
    
    # 2. Binary search for the minimal scope
    # Lower bound: simple counting bound is roughly total_diffs / (k+1) ? No, counting bound is sqrt-like or linear.
    # The prompt says counting bound is 112 for (4,7).
    # Let's start low and increase.
    
    # A better lower bound:
    # Each row has k*(k+1)/2 differences.
    # Total differences = n_rows * k*(k+1)/2.
    # These must be distinct integers in [1, M].
    # So M >= n_rows * k*(k+1)/2.
    # For (4,7): 4 * 28 = 112. This matches the prompt.
    
    lb = n_rows * diffs_per_row
    ub = current_best_scope
    
    # We can also try a slightly lower bound based on the max difference in a row.
    # The max difference in a row is the last element.
    # If the last element is M, the differences are in [1, M].
    
    # Let's try values from lb to ub.
    # Since the search space grows rapidly, we should be careful.
    # Start from lb and increment.
    
    found_scope = None
    found_rows = None
    
    # To handle potential timeouts in tight scopes, we allow multiple restarts for a given scope.
    # If we fail for a scope, we increase the scope.
    
    scope = lb
    while scope <= ub:
        if time.time() > deadline - 10:
            break
        
        # Try to find a solution with this scope
        # We might need multiple attempts for the same scope if the first one fails due to bad luck
        attempts = 0
        max_attempts_per_scope = 5 if scope < current_best_scope else 20
        
        while attempts < max_attempts_per_scope:
            if time.time() > deadline - 5:
                break
            
            # Use a different seed for each attempt to explore different parts of the space
            attempt_rng = random.Random(int(time.time() * 1000) + scope + attempts)
            
            res = csp_backtrack(n_rows, k, scope, attempt_rng)
            if res:
                found_scope = scope
                found_rows = res
                break
            attempts += 1
        
        if found_rows:
            break
        
        scope += 1
    
    if found_rows:
        best_rows = found_rows
        best_scope = found_scope
    else:
        # If we didn't find anything better, return the initial greedy solution
        best_rows = init_rows
        best_scope = current_best_scope

    # 3. Local optimization to try to reduce scope further if we have time
    # The CSP search is exhaustive for a fixed scope (with restarts), but it might find a solution
    # that isn't optimal in terms of "compactness" if we allow larger values.
    # However, since we binary searched/incremented scope, the first valid scope is the minimum.
    # Wait, the CSP search finds *a* solution with max entry <= scope.
    # If we find one at scope S, S is the minimum possible?
    # Not necessarily. We might have failed to find a solution at S-1 due to search limitations (timeouts, bad randomization).
    # But if we find one at S, we can try to compact it to see if it fits in S-1.
    
    if time.time() < deadline - 10:
        # Try to compact the best found rows
        # Simple local search: try to reduce the largest element
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_rows)):
                row = best_rows[i]
                # Try to reduce the last element
                if len(row) > 1:
                    orig_last = row[-1]
                    # Try values smaller than orig_last
                    for new_last in range(orig_last - 1, row[-2], -1):
                        test_row = row[:-1] + [new_last]
                        # Check validity
                        test_diffs = set()
                        valid = True
                        for a in test_row:
                            for b in test_row:
                                if a < b:
                                    d = b - a
                                    if d in test_diffs:
                                        valid = False
                                        break
                                    test_diffs.add(d)
                            if not valid:
                                break
                        
                        if valid:
                            # Check against other rows
                            conflict = False
                            for j in range(len(best_rows)):
                                if j == i:
                                    continue
                                other_diffs = get_diffs(best_rows[j])
                                for d in test_diffs:
                                    if d in other_diffs:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            
                            if not conflict:
                                best_rows[i] = test_row
                                improved = True
                                break # Restart compaction from the top
    
    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
