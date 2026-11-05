import time
import random
import math
import bisect
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    # The task is for (7,5) but we should handle general cases
    # For (7,5): 7 rows of 6 integers each
    # Best known: 112, counting bound: 105
    
    # Strategy: Use a constraint-based backtracking search with good heuristics
    # combined with local search for improvement
    
    # First, try to find a solution with scope as small as possible
    # using iterative deepening on the scope
    
    # Lower bound from counting: n_rows * (k+1)*k//2 differences must fit in [1, scope]
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    # So scope >= lower_bound (since differences are 1..scope, need at least lower_bound distinct values)
    # Actually, the differences range from 1 to scope, so we need scope >= lower_bound
    
    # For (7,5): lower_bound = 7 * 15 = 105
    
    # We'll use a hybrid approach:
    # 1. Try to construct solutions with increasing scope targets
    # 2. Use backtracking with constraint propagation
    # 3. Use local search to polish
    
    best_rows = None
    best_scope = float('inf')
    
    # Try scopes from lower_bound upward
    # But we need to be efficient - use a smarter search
    
    # Phase 1: Use a specialized backtracking solver with constraint propagation
    # This is the main structural change - using CSP-style search instead of pure greedy+SA
    
    # For the specific case of (7,5), we can be more targeted
    # Let's try a range of scope values
    
    # First, get a quick initial solution with greedy
    initial = _greedy_fast(n_rows, k, seed=0)
    if initial is not None:
        best_rows = initial
        best_scope = _scope(initial)
    
    # Now try to improve using backtracking with constraint propagation
    # Try scope values from best_scope-1 downward to lower_bound
    # But that's too many - use a smarter approach
    
    # Instead, use the backtracking solver with a target scope
    # and try several target scopes
    
    target_scopes = []
    if best_scope > lower_bound:
        # Try a few target scopes
        for s in range(max(lower_bound, best_scope - 20), best_scope):
            target_scopes.append(s)
    else:
        target_scopes = [lower_bound]
    
    # Also include some values around the known best (112 for 7,5)
    if n_rows == 7 and k == 5:
        for s in [105, 108, 110, 112, 114, 116]:
            if s not in target_scopes and s >= lower_bound:
                target_scopes.append(s)
        target_scopes = sorted(set(target_scopes))
    
    # Try each target scope with backtracking
    for target in target_scopes:
        if time.time() > deadline - 30:
            break
        
        result = _backtracking_solve(n_rows, k, target, start_time, deadline)
        if result is not None:
            result_scope = _scope(result)
            if result_scope < best_scope:
                best_scope = result_scope
                best_rows = result
                # If we hit the lower bound, we're done
                if best_scope == lower_bound:
                    break
    
    # Phase 2: Local search to further improve
    if best_rows is not None and time.time() < deadline - 10:
        improved = _local_search_polish(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Final targeted improvement on the max element
    if best_rows is not None and time.time() < deadline - 3:
        polished = _polish_max_element(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    if best_rows is None:
        # Fallback: return a valid solution
        best_rows = _greedy_fast(n_rows, k, seed=0)
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _greedy_fast(n_rows, k, seed=0):
    """Fast greedy construction to get an initial valid solution."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            used.update(new_diffs)
        rows.append(row)
    
    return rows

def _backtracking_solve(n_rows, k, target_scope, start_time, deadline):
    """
    Constraint-based backtracking search.
    Tries to construct n_rows rows of k+1 increasing integers starting at 0,
    with all differences distinct, and max element <= target_scope.
    """
    # We'll use a recursive backtracking with constraint propagation
    # State: current rows, used differences set
    
    # For efficiency, we'll build rows one at a time
    # Within each row, we choose elements sequentially
    
    # Precompute: for a given row prefix, what are the valid next values?
    
    def solve_recursive(rows, used_diffs, row_idx, elem_idx, current_row):
        """
        Recursive backtracking.
        rows: list of completed rows
        used_diffs: set of all differences used so far
        row_idx: which row we're building (0-indexed)
        elem_idx: which element in the current row we're choosing (0 is always 0)
        current_row: the current row being built
        """
        if time.time() > deadline - 25:
            return None
        
        if row_idx == n_rows:
            return rows
        
        if elem_idx == 0:
            # Start new row with 0
            return solve_recursive(rows, used_diffs, row_idx, 1, [0])
        
        if elem_idx == k + 1:
            # Row complete
            new_rows = rows + [current_row]
            return solve_recursive(new_rows, used_diffs, row_idx + 1, 0, [])
        
        # Choose the next element
        last = current_row[-1]
        # Try values from last+1 to target_scope
        # Use a heuristic: try smaller values first (to keep scope small)
        
        candidates = []
        for x in range(last + 1, target_scope + 1):
            # Check if x creates any duplicate differences
            valid = True
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                # Also check internal consistency (should be fine since row is increasing)
                candidates.append((x, new_diffs))
        
        if not candidates:
            return None
        
        # Try candidates - use a heuristic ordering
        # Prefer candidates that leave more room for future elements
        # and whose differences are "spread out" to avoid blocking future rows
        
        # Score candidates: prefer smaller x, and differences that are less "crowded"
        def score_candidate(x, new_diffs):
            # Prefer smaller x
            base = x
            # Penalize if new_diffs are close to each other (more flexible)
            if len(new_diffs) > 1:
                sorted_diffs = sorted(new_diffs)
                gaps = [sorted_diffs[i+1] - sorted_diffs[i] for i in range(len(sorted_diffs)-1)]
                min_gap = min(gaps) if gaps else 0
                base += max(0, 5 - min_gap)  # small penalty for tight gaps
            return base
        
        candidates.sort(key=lambda t: score_candidate(t[0], t[1]))
        
        # Limit the number of candidates we try to avoid exponential blowup
        # Use a beam-like approach: try top N candidates
        max_candidates = 20
        for x, new_diffs in candidates[:max_candidates]:
            new_row = current_row + [x]
            new_used = used_diffs | new_diffs
            
            result = solve_recursive(rows, new_used, row_idx, elem_idx + 1, new_row)
            if result is not None:
                return result
        
        return None
    
    # Start the search
    result = solve_recursive([], set(), 0, 0, [])
    return result

def _local_search_polish(rows, n_rows, k, start_time, deadline):
    """
    Local search: try to replace individual rows with better ones.
    """
    rng = random.Random(99)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute row differences
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    max_val = max(row[-1] for row in rows)
    
    # Try to improve each row
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        # Pick a random row to try to improve
        row_idx = rng.randint(0, n_rows - 1)
        current_max = rows[row_idx][-1]
        
        # Collect differences from other rows
        used_by_others = set()
        for i in range(n_rows):
            if i != row_idx:
                used_by_others.update(row_diffs[i])
        
        # Try to find a better row
        new_row = _find_better_row(rows, row_idx, k, rng, used_by_others, current_max)
        
        if new_row is not None and new_row[-1] < current_max:
            # Check validity
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - row[i])
            
            if not (new_diffs & used_by_others):
                # Valid improvement
                rows[row_idx] = new_row
                row_diffs[row_idx] = new_diffs
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _find_better_row(rows, row_idx, k, rng, used_by_others, current_max):
    """Try to find a replacement row with a smaller maximum."""
    target_max = current_max - 1
    
    for attempt in range(10):
        row = [0]
        local_used = set(used_by_others)
        success = True
        
        for _ in range(k):
            last = row[-1]
            valid_xs = []
            
            for x in range(last + 1, target_max + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used):
                    valid_xs.append(x)
            
            if not valid_xs:
                success = False
                break
            
            # Choose from valid candidates
            # Prefer smaller x for tighter packing
            valid_xs.sort()
            # Pick from the smaller ones
            top_n = min(5, len(valid_xs))
            chosen = valid_xs[rng.randint(0, top_n - 1)]
            
            row.append(chosen)
            new_diffs = {chosen - a for a in row[:-1]}
            local_used.update(new_diffs)
        
        if success and len(row) == k + 1:
            return row
        
        # Relax target
        target_max = min(target_max + 1, current_max - 1)
    
    return None

def _polish_max_element(rows, n_rows, k, start_time, deadline):
    """
    Targeted polish: repeatedly try to reduce the overall maximum element.
    """
    rng = random.Random(77)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        # Find the row with the maximum element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to reduce this row's maximum
        # Collect differences from other rows
        used_by_others = set()
        for i in range(n_rows):
            if i != max_idx:
                n = len(rows[i])
                for a in range(n):
                    for b in range(a + 1, n):
                        used_by_others.add(rows[i][b] - rows[i][a])
        
        # Try to find a better version of this row
        new_row = _find_better_row(rows, max_idx, k, rng, used_by_others, current_max)
        
        if new_row is not None and new_row[-1] < current_max:
            # Check validity
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            
            if not (new_diffs & used_by_others):
                rows[max_idx] = new_row
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
    
    return best_rows if best_scope < _scope([list(r) for r in rows]) else None

def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2
