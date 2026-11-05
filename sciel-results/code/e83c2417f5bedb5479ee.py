import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit

    best_rows = None
    best_score = float('inf')

    expected_diffs = (k + 1) * k // 2

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        return len(diffs) == expected_diffs

    # Backtracking search
    # State: current row being built, list of completed rows, set of used differences
    # Heuristic: Try smaller values first (greedy), but backtrack if stuck.
    # To improve efficiency, we can use a "cost" for differences.
    # Differences that are small are more "valuable" because they appear in many potential rows?
    # Actually, large differences are more constrained in a small scope?
    # Let's stick to a robust backtracking with randomized restarts and a time limit.
    
    def backtrack_search(max_scope):
        """
        Attempts to find a valid configuration with scope <= max_scope.
        Returns the list of rows if found, None otherwise.
        """
        used_diffs = set()
        rows = []
        
        # We build rows one by one.
        # For each row, we build it element by element.
        
        def build_row(row_idx):
            if row_idx == n_rows:
                return True
            
            # Heuristic: Start with a smaller row? Or randomize row order?
            # For now, build rows in order.
            # To break symmetry, we can fix the first element to 0 (already done).
            
            # Backtracking state for current row
            current_row = [0]
            
            def extend_row(pos):
                if pos == k + 1:
                    # Row complete
                    rows.append(list(current_row))
                    if build_row(row_idx + 1):
                        return True
                    rows.pop()
                    # Remove diffs of this row from used_diffs
                    for d in row_diffs(current_row):
                        used_diffs.discard(d)
                    return False
                
                last_val = current_row[-1]
                # Try values from last_val + 1 to max_scope
                # Heuristic: Smaller values are generally better for minimizing scope
                # But we can also randomize the order slightly to escape local minima
                # Let's try a mix: mostly greedy, but with some randomization in restarts
                
                # To make backtracking effective, we should prune early.
                # A simple pruning: if the remaining elements cannot be placed without exceeding max_scope?
                # This is hard to estimate.
                
                # Let's iterate values
                # Optimization: If we are close to the limit, be more selective?
                
                for val in range(last_val + 1, max_scope + 1):
                    # Check if val can be added
                    new_diffs = {val - a for a in current_row}
                    if new_diffs & used_diffs:
                        continue
                    if len(new_diffs) != len(current_row): # Internal duplicates
                        continue
                    
                    # Add val
                    current_row.append(val)
                    used_diffs.update(new_diffs)
                    
                    if extend_row(pos + 1):
                        return True
                    
                    # Backtrack
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            
            return extend_row(1)

        return build_row(0)

    # Main loop: Try increasing scopes or randomized restarts with fixed scope?
    # The counting bound is 105. Best known is 112.
    # Let's try to find a solution with scope 105, 106, ...
    # Or use randomized greedy to get a good initial scope, then backtrack to improve?
    
    # Strategy:
    # 1. Quick randomized greedy to get an initial bound (fast).
    # 2. Backtracking search to try to beat that bound, starting from bound-1 down to counting bound.
    
    # 1. Randomized Greedy (fast, to get a ballpark)
    def randomized_greedy(max_scope_hint=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score - 1) # Try to beat current best
                
                valids = []
                last = row[-1]
                for x in range(last + 1, cap + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        valids.append(x)
                
                if not valids:
                    return None

                # Weighted random choice, favoring smaller x
                if len(valids) > 1:
                    consider = valids[:20]
                    weights = [1.0 / (v + 1) for v in consider]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = consider[0]
                    for v, w in zip(consider, weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = valids[0]

                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    # Run a few quick greedy attempts to establish an initial best_score
    for _ in range(10):
        if time.time() - start_time > time_limit - 5:
            break
        rows = randomized_greedy()
        if rows:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]

    # 2. Backtracking search to improve
    # Try scopes from best_score - 1 down to 105 (or counting bound)
    # We need to be careful with time.
    
    # To make backtracking faster, we can use a "most constrained" heuristic for value ordering.
    # But simple value ordering (ascending) is often good for minimizing max.
    # Let's add a time check inside the backtracking loop.
    
    def backtrack_search_timed(max_scope):
        used_diffs = set()
        rows = []
        
        def build_row(row_idx):
            if time.time() - start_time > time_limit - 2:
                return False # Timeout
            if row_idx == n_rows:
                return True
            
            current_row = [0]
            
            def extend_row(pos):
                if time.time() - start_time > time_limit - 2:
                    return False
                if pos == k + 1:
                    rows.append(list(current_row))
                    if build_row(row_idx + 1):
                        return True
                    rows.pop()
                    for d in row_diffs(current_row):
                        used_diffs.discard(d)
                    return False
                
                last_val = current_row[-1]
                # Heuristic: Try smaller values first
                # To avoid getting stuck in a single deep path, we can add some randomness
                # but for "exact" search, deterministic is better for finding *a* solution.
                # However, different row orderings might help.
                
                # Pruning: If we have used too many small differences, it might be hard to complete?
                # Hard to quantify.
                
                for val in range(last_val + 1, max_scope + 1):
                    new_diffs = {val - a for a in current_row}
                    if new_diffs & used_diffs:
                        continue
                    if len(new_diffs) != len(current_row):
                        continue
                    
                    current_row.append(val)
                    used_diffs.update(new_diffs)
                    
                    if extend_row(pos + 1):
                        return True
                    
                    current_row.pop()
                    used_diffs.difference_update(new_diffs)
                
                return False
            
            return extend_row(1)

        if build_row(0):
            return rows
        return None

    # Try to beat the current best score
    # Start from best_score - 1 and go down
    # If best_score is inf, start from a reasonable estimate (e.g., 120)
    
    current_target = best_score - 1 if best_score < float('inf') else 120
    
    # Ensure we don't go below the theoretical minimum
    # Counting bound: n_rows * (k+1)*k/2 <= Scope*(Scope+1)/2 approx
    # Scope ~ sqrt(2 * n_rows * expected_diffs)
    min_scope = int((2 * n_rows * expected_diffs) ** 0.5)
    
    while current_target >= min_scope and time.time() - start_time < time_limit - 5:
        # Randomize the search a bit by shuffling row construction order?
        # The backtracking builds rows 0..n_rows-1.
        # We can try different permutations of row indices?
        # No, rows are indistinguishable except for their content.
        # But the *order* in which we assign them can affect the search tree.
        
        # Let's just run the deterministic backtracking.
        # To avoid repeated failures on the same target, we might need to add randomness
        # or change the heuristic.
        
        # For now, let's just try it. If it fails, we might need a more sophisticated heuristic.
        # A common improvement is to use "Minimum Remaining Values" (MRV) for the *next row*?
        # Or for the *next value*?
        
        # Let's add a simple randomization: pick a random subset of valid values to try first?
        # No, that breaks the "smallest scope" guarantee of the search order.
        
        # Instead, let's rely on the fact that if greedy found a solution at S,
        # backtracking might find one at S-1.
        
        rows = backtrack_search_timed(current_target)
        if rows:
            best_score = current_target
            best_rows = rows
            current_target -= 1
        else:
            # If it fails, try a slightly higher target? Or break?
            # If it fails at T, it might fail at T-1 too.
            # But it's not guaranteed.
            # Let's try T-1.
            current_target -= 1
            
            # If we've tried a few and failed, maybe we should stop?
            # Or we can randomize the initial greedy to get a better starting point?
            # Let's just continue until time runs out.

    # If no improvement was found via backtracking, return the greedy result
    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
