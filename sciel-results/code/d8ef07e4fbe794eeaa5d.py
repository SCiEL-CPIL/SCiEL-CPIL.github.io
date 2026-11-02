import time
import random
import math
import heapq

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences are distinct)."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _generate_sidon_rows(k, max_scope, target_count):
    """
    Generate a list of valid Sidon rows of length k+1 with max value <= max_scope.
    Returns list of (max_val, row) sorted by max_val.
    """
    rows = []
    # Use a recursive generator or iterative with pruning
    # Given small k (5) and max_scope (~120), we can be somewhat brute force with pruning
    
    def backtrack(current_row, start_val, used_diffs):
        if len(current_row) == k + 1:
            rows.append((current_row[-1], list(current_row)))
            return
        
        if len(rows) >= target_count:
            return
            
        # Pruning: if we've found enough, stop
        # Estimate remaining space
        remaining_slots = (k + 1) - len(current_row)
        
        # Try next values
        # Optimization: start from smallest possible
        # We need to ensure we can fit remaining_slots numbers with distinct diffs
        
        last_val = current_row[-1] if current_row else 0
        
        # Heuristic: Don't go too high if we already have many rows
        # But we need to find small ones first.
        
        for v in range(last_val + 1, max_scope + 1):
            # Check if v creates duplicate diffs with current_row
            new_diffs = set()
            valid = True
            for a in current_row:
                d = v - a
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Additional pruning: Can we fit the rest?
            # The smallest possible next value is v+1.
            # The differences will be at least 1, 2, etc.
            # This is hard to prune tightly without more logic.
            
            current_row.append(v)
            used_diffs.update(new_diffs)
            
            backtrack(current_row, v, used_diffs)
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
            if len(rows) >= target_count:
                return

    backtrack([], 0, set())
    rows.sort(key=lambda x: x[0])
    return [r for _, r in rows[:target_count]]

def _greedy_init(n_rows, k, scope_limit):
    """Greedy initialization: build rows one by one, choosing smallest valid values."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            found = False
            for x in range(row[-1] + 1, scope_limit + 1):
                new_diffs = {x - a for a in row}
                # Check internal distinctness
                if len(new_diffs) != len(row):
                    continue
                # Check external distinctness
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _global_max(rows):
    """Return the maximum value across all rows."""
    return max(max(r) for r in rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate a pool of candidate rows
    # We need n_rows disjoint rows.
    # Let's generate a decent number of small rows.
    # Scope 112 is the target. Let's generate rows up to 120.
    # Generating all Sidon rows up to 120 might be slow if done naively.
    # But k=5 is small.
    
    # Let's try generating 500 smallest Sidon rows.
    # If that's not enough, we generate more or increase scope.
    
    candidates = []
    current_scope_limit = 80
    
    while time.time() < deadline - 10:
        # Generate candidates up to current_scope_limit
        new_candidates = _generate_sidon_rows(k, current_scope_limit, 1000)
        
        if not new_candidates:
            current_scope_limit += 10
            continue
            
        # Merge and sort
        # To avoid duplicates in candidates, we can use a set of tuples
        existing = set(tuple(r) for r in candidates)
        for r in new_candidates:
            if tuple(r) not in existing:
                candidates.append(r)
                existing.add(tuple(r))
                
        candidates.sort(key=lambda r: r[-1])
        
        # Try to solve using backtracking on these candidates
        # We want to pick n_rows rows with disjoint diffs.
        
        best_solution = None
        best_scope = float('inf')
        
        # Use a recursive backtracking search
        # To speed up, we can use a greedy heuristic with backtracking.
        
        def solve(idx, used_diffs, current_rows):
            nonlocal best_solution, best_scope
            
            if time.time() > deadline - 5:
                return
                
            if idx == n_rows:
                scope = _global_max(current_rows)
                if scope < best_scope:
                    best_scope = scope
                    best_solution = [list(r) for r in current_rows]
                return
                
            # Pruning: if the smallest possible scope for remaining rows exceeds best_scope, stop?
            # Hard to estimate.
            
            # Try candidates
            # Optimization: Only try candidates that are "small enough"
            # And skip if they conflict
            
            # To avoid trying the same candidates repeatedly in different orders,
            # we can enforce an order: row[i] must be "larger" or "different" in some way?
            # No, rows are indistinguishable in the set.
            # We can enforce that we pick from candidates in a sorted order and don't reuse indices?
            # Yes, since we just need a set of rows, we can assign them to indices 0..n_rows-1.
            # To avoid symmetry breaking issues, we can just pick a subset.
            
            # However, backtracking over 1000 candidates for 7 rows is 1000^7 which is impossible.
            # We need a stronger heuristic.
            
            # Heuristic: Pick the candidate with the smallest max value that doesn't conflict.
            # If that fails, try the next smallest, etc.
            # This is essentially a greedy algorithm with backtracking.
            
            # To make it efficient, we can limit the branching factor.
            # Or use a Beam Search.
            
            # Let's try a simple greedy with limited backtracking (e.g., try top 50 candidates at each step).
            
            # Sort candidates by max value
            # We already have them sorted.
            
            # We need to track which candidates have been used?
            # Since rows are unique, we can just pick distinct rows from the pool.
            
            # Let's iterate through candidates
            # To avoid O(N^k) complexity, we rely on the fact that small rows are rare and conflicts are common.
            
            # A better approach: 
            # 1. Pick the smallest row.
            # 2. Pick the next smallest row that doesn't conflict.
            # 3. If stuck, backtrack to the previous choice and try the next alternative.
            
            # This is a standard DFS.
            # To prune, we can check if the current max value already exceeds best_scope.
            
            current_max = max(r[-1] for r in current_rows) if current_rows else 0
            
            for i, row in enumerate(candidates):
                if row[-1] > best_scope:
                    break # Since sorted, no more candidates can improve best_scope
            
                # Check conflict
                row_diffs = set()
                valid = True
                for j in range(len(row)):
                    for l in range(j + 1, len(row)):
                        d = row[l] - row[j]
                        if d in used_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                if not valid:
                    continue
                
                # Additional check: internal distinctness (should be true by generator, but just in case)
                if len(row_diffs) != len(row) * (len(row) - 1) // 2:
                    continue
                
                # Recurse
                current_rows.append(row)
                used_diffs.update(row_diffs)
                
                solve(idx + 1, used_diffs, current_rows)
                
                current_rows.pop()
                used_diffs.difference_update(row_diffs)
                
                # If we found a solution, we might want to try to find a better one?
                # The loop continues to try other candidates to find smaller scope.
                # But if we found one, best_scope is updated.
                # The condition `if row[-1] > best_scope: break` will prune subsequent larger rows.
                
                # Optimization: If we found a solution with scope S, we can stop trying rows with max > S.
                # This is already handled by the break.
                
                # To speed up, if we find a solution, we can reduce the search space?
                # Not easily.
                
                # Limit branching: If we've tried too many, stop?
                # No, we need to find the optimal.
                
                # However, 1000 candidates is too many for depth 7.
                # We need to reduce the pool or use a smarter search.
                
                # Let's try to reduce the pool to the top 100 smallest rows.
                # If 100 isn't enough, we increase the limit.
                
            return

        # Run the solver
        # To make it faster, we can use a greedy initialization first to get a good upper bound (best_scope).
        
        init_rows = _greedy_init(n_rows, k, current_scope_limit + 20)
        if init_rows:
            best_scope = _global_max(init_rows)
            best_solution = init_rows
            
        # Now run DFS with this best_scope as a bound
        solve(0, set(), [])
        
        if best_solution:
            return best_solution
            
        # If no solution found, increase scope limit and retry
        current_scope_limit += 10
        
    # Fallback
    return _greedy_init(n_rows, k, 300)
