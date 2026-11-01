import time
import random
import sys
from collections import defaultdict

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0  # safety margin before 150s kill

    # Number of differences per row
    d_per_row = k * (k + 1) // 2
    total_diffs = n_rows * d_per_row

    # Counting lower bound
    m_min = 0
    while m_min * (m_min + 1) // 2 < total_diffs:
        m_min += 1
    lower_bound = m_min

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    # ========================================================================
    # CSP-style backtracking with scope bounding
    # ========================================================================
    
    def solve_with_bound(M, time_budget=3.0):
        """
        Try to construct n_rows rows of k+1 increasing integers starting at 0,
        all entries <= M, with all differences distinct across all rows.
        Uses backtracking with forward checking.
        Returns list of rows if found, else None.
        """
        start_time = time.time()
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, pos_idx, current_row):
            if time.time() - start_time > time_budget:
                return False
            
            if pos_idx == k + 1:
                rows.append(current_row[:])
                if row_idx + 1 == n_rows:
                    return True
                return backtrack(row_idx + 1, 1, [0])
            
            lo = current_row[-1] + 1
            hi = M
            
            if hi <= lo:
                return False
            
            remaining_positions = k + 1 - pos_idx - 1
            hi_adjusted = min(hi, M - remaining_positions)
            
            if hi_adjusted < lo:
                return False
            
            candidates = list(range(lo, hi_adjusted + 1))
            
            if len(candidates) > 50:
                n_try = min(len(candidates), 30)
                sampled = set()
                for v in candidates[:min(10, len(candidates))]:
                    sampled.add(v)
                for _ in range(max(0, n_try - len(sampled))):
                    sampled.add(rng.choice(candidates))
                candidates = sorted(sampled)
            
            for val in candidates:
                new_diffs = set()
                valid = True
                for prev_val in current_row:
                    diff = val - prev_val
                    if diff in used_diffs or diff in new_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                
                if not valid:
                    continue
                
                current_row.append(val)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos_idx + 1, current_row):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if backtrack(0, 1, [0]):
            return rows
        return None

    # ========================================================================
    # Targeted descent
    # ========================================================================
    
    def targeted_descent(rows, current_best_scope, deadline):
        """Try to reduce scope by shrinking the largest mark in rows with max."""
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False
        start = time.time()
        
        while time.time() < min(deadline - 5, start + 10):
            max_val = max(max(r) for r in cur)
            if max_val <= current_best_scope:
                break
            
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1
            
            if old_val <= lower:
                continue
            
            candidates = set()
            for v in range(lower, min(lower + 15, old_val)):
                candidates.add(v)
            for _ in range(10):
                candidates.add(rng.randint(lower, old_val - 1))
            
            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate_scope(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val
            
            if not improved:
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(5):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate_scope(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
        
        if improved and cur_scope < current_best_scope:
            return cur
        return None

    # ========================================================================
    # Simulated annealing
    # ========================================================================
    
    def run_simulated_annealing(initial_rows, initial_scope, deadline):
        """Run SA and return (best_rows_found, best_scope_found)."""
        max_restarts = 50
        steps_per_restart = 50000
        
        sa_best_rows = [r[:] for r in initial_rows]
        sa_best_scope = initial_scope
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                cur = [r[:] for r in initial_rows]
            else:
                cur = [r[:] for r in sa_best_rows]
                for _ in range(rng.randint(1, 3)):
                    ri = rng.randrange(n_rows)
                    pos = rng.randint(1, len(cur[ri]) - 1)
                    lo = cur[ri][pos - 1] + 1
                    hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else sa_best_scope + 10
                    if hi > lo:
                        cur[ri][pos] = rng.randint(lo, hi)
            
            cur_scope = full_validate_scope(cur)
            if cur_scope is None:
                continue
            
            if cur_scope < sa_best_scope:
                sa_best_scope = cur_scope
                sa_best_rows = [r[:] for r in cur]
            
            T = max(0.05, 1.0 - restart / max_restarts)
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                old_val = cur[ri][pos]
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else sa_best_scope + 20
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                
                cur[ri][pos] = new_val
                new_scope = full_validate_scope(cur)
                
                if new_scope is not None:
                    if new_scope < sa_best_scope:
                        sa_best_scope = new_scope
                        sa_best_rows = [r[:] for r in cur]
                    delta = new_scope - cur_scope
                    if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                        cur_scope = new_scope
                    else:
                        cur[ri][pos] = old_val
                else:
                    cur[ri][pos] = old_val
                
                T *= 0.99995
            
            if time.time() < deadline - 5:
                desc_result = targeted_descent(cur, sa_best_scope, deadline)
                if desc_result is not None:
                    new_scope = get_scope(desc_result)
                    if new_scope < sa_best_scope:
                        sa_best_scope = new_scope
                        sa_best_rows = [r[:] for r in desc_result]
        
        return sa_best_rows, sa_best_scope

    # ========================================================================
    # Main search: iterative deepening on scope bound
    # ========================================================================
    
    greedy_scope = get_scope(greedy_init())
    
    search_bounds = set()
    start_bound = max(lower_bound, best_scope - 20)
    
    for m in range(start_bound, min(start_bound + 15, best_scope + 5) + 1):
        search_bounds.add(m)
    search_bounds.add(best_scope)
    search_bounds.add(best_scope + 1)
    search_bounds.add(best_scope + 2)
    
    sorted_bounds = sorted(search_bounds)
    
    found_csp_solution = False
    for M in sorted_bounds:
        if time.time() > deadline - 25:
            break
        
        if M < lower_bound:
            continue
        
        result = solve_with_bound(M, time_budget=3.0)
        
        if result is not None:
            scope = get_scope(result)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in result]
            
            if time.time() < deadline - 15:
                improved = targeted_descent(result, best_scope, deadline)
                if improved is not None:
                    new_scope = get_scope(improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in improved]
            
            found_csp_solution = True
            break
    
    # ========================================================================
    # If time remains, run simulated annealing
    # ========================================================================
    
    if time.time() < deadline - 10:
        sa_rows, sa_scope = run_simulated_annealing(best_rows, best_scope, deadline)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
    
    # Final fallback
    if best_rows is None:
        best_rows = greedy_init()
    
    return best_rows
