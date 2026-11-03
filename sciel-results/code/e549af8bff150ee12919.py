"""Difference triangle set (4,7) — ILP formulation with greedy fallback and local search."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diff_set(row):
    """Compute the set of all positive differences in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diff_set(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _enumerate_sidon_rows(k, max_scope, time_limit=30.0):
    """
    Enumerate all valid Sidon rows of length k+1 starting at 0 with max entry <= max_scope.
    A row is valid if all its pairwise differences are distinct (within the row).
    Returns a list of (row_tuple, diff_set, max_entry).
    """
    start_time = time.time()
    results = []
    needed_len = k + 1
    
    # Use backtracking to enumerate all valid rows
    def backtrack(current_row, used_diffs):
        if time.time() - start_time > time_limit:
            return
        if len(current_row) == needed_len:
            results.append((tuple(current_row), set(used_diffs), current_row[-1]))
            return
        
        last = current_row[-1] if current_row else 0
        # Try next values starting from last + 1
        for next_val in range(last + 1, max_scope + 1):
            # Check if adding next_val creates duplicate differences within the row
            new_diffs = set()
            valid = True
            for a in current_row:
                d = next_val - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                current_row.append(next_val)
                used_diffs |= new_diffs
                backtrack(current_row, used_diffs)
                used_diffs -= new_diffs
                current_row.pop()
    
    backtrack([0], set())
    return results


def _solve_ilp(n_rows, k, patterns, deadline):
    """
    Solve the ILP: select n_rows disjoint patterns (no shared differences).
    Objective: minimize the maximum scope among selected patterns.
    Uses scipy.optimize.milp if available, otherwise falls back to greedy selection.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return None
    
    num_patterns = len(patterns)
    if num_patterns == 0:
        return None
    
    # Get all unique differences
    all_diffs = set()
    for _, diff_set, _ in patterns:
        all_diffs |= diff_set
    all_diffs_list = sorted(all_diffs)
    diff_to_idx = {d: i for i, d in enumerate(all_diffs_list)}
    num_diffs = len(all_diffs_list)
    
    # Build constraint matrix: for each difference, sum of patterns using it <= 1
    # A is (num_diffs, num_patterns)
    A = np.zeros((num_diffs, num_patterns), dtype=np.int8)
    for j, (_, diff_set, _) in enumerate(patterns):
        for d in diff_set:
            A[diff_to_idx[d], j] = 1
    
    # Objective: minimize max scope. We use a binary search on M.
    # For a given M, check if we can select n_rows patterns with scope <= M.
    
    # Get scopes
    scopes = [p[2] for p in patterns]
    
    # Binary search on M
    low_m = min(scopes) if scopes else 0
    high_m = max(scopes) if scopes else 0
    
    # Try to find the minimum M such that we can select n_rows disjoint patterns with scope <= M
    best_solution = None
    
    # Filter patterns by scope for each candidate M
    # To speed up, sort patterns by scope
    sorted_indices = sorted(range(num_patterns), key=lambda i: scopes[i])
    
    # Try a few candidate M values: the sorted unique scopes
    unique_scopes = sorted(set(scopes))
    
    # Limit the number of ILP solves to avoid timeout
    max_ilp_solves = 15
    candidates_to_try = unique_scopes[:max_ilp_solves]
    
    for M in candidates_to_try:
        if time.time() > deadline - 5:
            break
        
        # Filter patterns with scope <= M
        valid_indices = [i for i in sorted_indices if scopes[i] <= M]
        if len(valid_indices) < n_rows:
            continue
        
        # Build reduced problem
        A_sub = A[:, valid_indices]
        num_vars = len(valid_indices)
        
        # Variables: binary, select n_rows patterns
        # Constraint: A_sub @ x <= 1 (each difference used at most once)
        #              sum(x) == n_rows
        
        # Full constraint matrix: [A_sub; ones(1, num_vars)]
        A_full = np.vstack([A_sub, np.ones((1, num_vars), dtype=np.int8)])
        b_full = np.concatenate([np.ones(num_diffs, dtype=np.int8), [n_rows]])
        
        # Objective: minimize sum of scopes (or just feasibility check)
        # For feasibility, c = 0
        c = np.zeros(num_vars)
        
        constraints = LinearConstraint(A_full, -np.inf, b_full)
        integrality = np.ones(num_vars)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={'time_limit': 5, 'mip_gap': 0.0})
            if res.status == 0 and res.success:
                x = res.x
                selected = [valid_indices[i] for i in range(num_vars) if x[i] > 0.5]
                if len(selected) == n_rows:
                    best_solution = [patterns[i][0] for i in selected]
                    break
        except Exception:
            continue
    
    if best_solution:
        # Convert to list of lists
        return [list(r) for r in best_solution]
    
    return None


def _greedy_select(n_rows, k, patterns, deadline):
    """
    Greedy selection of n_rows disjoint patterns minimizing max scope.
    Sorts patterns by scope and greedily picks non-conflicting ones.
    """
    # Sort by scope
    sorted_patterns = sorted(patterns, key=lambda p: p[2])
    
    selected = []
    used_diffs = set()
    
    for row, diff_set, scope in sorted_patterns:
        if time.time() > deadline:
            break
        if len(selected) >= n_rows:
            break
        if not (diff_set & used_diffs):
            selected.append(list(row))
            used_diffs |= diff_set
    
    if len(selected) == n_rows:
        return selected
    
    return None


def _local_search(rows, deadline, rng):
    """
    Local search: try to reduce the maximum entry by adjusting individual
    row elements while maintaining validity.
    """
    best = [list(r) for r in rows]
    best_score = _scope(best)
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random row to modify
        row_idx = rng.randrange(len(best))
        old_row = best[row_idx][:]
        
        # Try to rebuild this row with a slightly different strategy
        used_others = set()
        for i, r in enumerate(best):
            if i != row_idx:
                used_others |= _row_diff_set(r)
        
        # Rebuild row with randomness
        new_row = [0]
        x = 0
        k_needed = len(old_row)
        while len(new_row) < k_needed:
            x += 1
            new = {x - a for a in new_row}
            if not new & used_others:
                if rng.random() < 0.7:
                    new_row.append(x)
                else:
                    if rng.random() < 0.5:
                        new_row.append(x)
        
        if len(new_row) == k_needed:
            trial_rows = [list(r) for r in best]
            trial_rows[row_idx] = new_row
            if _is_valid(trial_rows):
                sc = _scope(trial_rows)
                if sc < current_score:
                    current_score = sc
                    best = trial_rows
                    if sc < best_score:
                        best_score = sc
                elif rng.random() < 0.1:
                    best = trial_rows
                    current_score = sc
    
    return best


def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Start with a good greedy baseline
    best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
    best_score = _scope(best)
    
    # Try multiple greedy variants
    for seed in range(1, 15):
        if time.time() > deadline - 30:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.9, 0.8]))
        sc = _scope(trial)
        if sc < best_score:
            best_score = sc
            best = trial
    
    # Enumerate Sidon rows for ILP
    # Start with a reasonable max_scope based on current best
    initial_max_scope = best_score + 10
    if initial_max_scope > 200:
        initial_max_scope = 200
    
    patterns = _enumerate_sidon_rows(k, initial_max_scope, time_limit=20.0)
    
    if patterns:
        # Try ILP
        ilp_result = _solve_ilp(n_rows, k, patterns, deadline - 20)
        if ilp_result and _is_valid(ilp_result):
            ilp_score = _scope(ilp_result)
            if ilp_score < best_score:
                best = ilp_result
                best_score = ilp_score
        
        # Also try greedy selection from patterns
        greedy_result = _greedy_select(n_rows, k, patterns, deadline - 10)
        if greedy_result and _is_valid(greedy_result):
            greedy_score = _scope(greedy_result)
            if greedy_score < best_score:
                best = greedy_result
                best_score = greedy_score
    
    # Local search to improve
    if time.time() < deadline - 5:
        best = _local_search(best, deadline - 2, rng)
        best_score = _scope(best)
    
    # If time remains, try more restarts
    while time.time() < deadline - 5.0:
        trial = _greedy_build(n_rows, k, rng, bias=rng.choice([1.0, 0.9, 0.8]))
        sc = _scope(trial)
        if sc < best_score:
            best_score = sc
            best = trial
            if time.time() < deadline - 10.0:
                best = _local_search(best, min(deadline - 5.0, time.time() + 10.0), rng)
                best_score = _scope(best)
    
    return best

# EVOLVE-BLOCK-END
