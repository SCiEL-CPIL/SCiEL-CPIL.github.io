import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1
    k_plus_1 = k + 1
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Difference-set based optimization
    best_solution = _optimize_by_diffsets(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Iterative difference swapping
    best_solution = _swap_improve(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 4. Final compaction
    best_solution = _final_compact(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    return best_solution


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


def _row_to_diffs(row):
    """Convert a row to its set of differences."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _diffs_to_row(diffs, k):
    """Try to reconstruct a row from a set of differences."""
    # The row must start at 0, so 0 is always present
    # We need to find positions such that all differences in the set are present
    # and no extra differences are created
    k_plus_1 = k + 1
    
    # Simple approach: greedily build the row
    row = [0]
    used_diffs = set()
    remaining = set(diffs)
    
    for _ in range(k):
        # Try to extend the row
        best_x = None
        best_score = float('inf')
        
        for x in range(1, max(diffs) + 1):
            if x in row:
                continue
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            # Check if new_diffs are subset of remaining
            if not new_diffs.issubset(remaining):
                continue
            # Score: prefer smaller x, and prefer x that uses more remaining diffs
            score = x - len(new_diffs & remaining) * 10
            if score < best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            return None
        
        row.append(best_x)
        new_diffs = {best_x - a for a in row[:-1]}
        used_diffs |= new_diffs
        remaining -= new_diffs
    
    # Verify
    if len(row) != k_plus_1:
        return None
    if _row_to_diffs(row) != diffs:
        return None
    return row


def _optimize_by_diffsets(solution, n_rows, k, deadline):
    """Optimize by working with difference sets directly."""
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Get difference sets for each row
    diff_sets = [_row_to_diffs(row) for row in best]
    
    # Try to swap differences between rows to reduce scope
    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                # Try to swap some differences between rows i and j
                diffs_i = diff_sets[i]
                diffs_j = diff_sets[j]
                
                # Find differences that are "expensive" (large) in one row
                # and could be moved to another row
                for d in list(diffs_i):
                    if d > best_scope // 2:  # Only consider large diffs
                        # Check if d could be in row j
                        # and if some diff from j could be in row i
                        for d2 in list(diffs_j):
                            if d2 < d:
                                # Try swap: move d to j, d2 to i
                                new_diffs_i = (diffs_i - {d}) | {d2}
                                new_diffs_j = (diffs_j - {d2}) | {d}
                                
                                row_i_new = _diffs_to_row(new_diffs_i, k)
                                row_j_new = _diffs_to_row(new_diffs_j, k)
                                
                                if row_i_new is not None and row_j_new is not None:
                                    new_scope = max(max(row_i_new), max(row_j_new),
                                                   max(max(r) for idx, r in enumerate(best) if idx not in [i, j]))
                                    if new_scope < best_scope:
                                        best[i] = row_i_new
                                        best[j] = row_j_new
                                        diff_sets[i] = new_diffs_i
                                        diff_sets[j] = new_diffs_j
                                        best_scope = new_scope
                                        improved = True
                                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Final pull to smallest
    best = _pull_to_smallest_all(best, n_rows, k, deadline)
    
    new_scope = max(max(row) for row in best)
    return best if new_scope <= best_scope else [list(row) for row in solution]


def _swap_improve(solution, n_rows, k, deadline):
    """Iterative improvement via difference swapping with simulated annealing."""
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Build global difference tracking
    all_diffs = set()
    for row in best:
        all_diffs |= _row_to_diffs(row)
    
    # Simulated annealing on difference swaps
    current = [list(row) for row in best]
    current_scope = best_scope
    
    start_temp = 5.0
    end_temp = 0.1
    total_steps = 200
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, total_steps - 1))
    
    for step in range(total_steps):
        if time.time() > deadline - 10.0:
            break
        
        temp = start_temp * (cooling_rate ** step)
        
        # Pick two random rows
        i, j = random.sample(range(n_rows), 2)
        
        # Pick a difference from row i
        diffs_i = _row_to_diffs(current[i])
        diffs_j = _row_to_diffs(current[j])
        
        if not diffs_i or not diffs_j:
            continue
        
        d_i = random.choice(list(diffs_i))
        d_j = random.choice(list(diffs_j))
        
        if d_i == d_j:
            continue
        
        # Try swap
        new_diffs_i = (diffs_i - {d_i}) | {d_j}
        new_diffs_j = (diffs_j - {d_j}) | {d_i}
        
        row_i_new = _diffs_to_row(new_diffs_i, k)
        row_j_new = _diffs_to_row(new_diffs_j, k)
        
        if row_i_new is not None and row_j_new is not None:
            # Check global validity
            other_diffs = set()
            for idx, row in enumerate(current):
                if idx not in [i, j]:
                    other_diffs |= _row_to_diffs(row)
            
            if not (new_diffs_i & new_diffs_j) and not (new_diffs_i & other_diffs) and not (new_diffs_j & other_diffs):
                new_scope = max(max(row_i_new), max(row_j_new),
                               max(max(r) for idx, r in enumerate(current) if idx not in [i, j]))
                delta = new_scope - current_scope
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    current[i] = row_i_new
                    current[j] = row_j_new
                    current_scope = new_scope
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [list(row) for row in current]
    
    # Final compaction
    best = _pull_to_smallest_all(best, n_rows, k, deadline)
    
    new_scope = max(max(row) for row in best)
    return best if new_scope <= best_scope else [list(row) for row in solution]


def _pull_to_smallest_all(solution, n_rows, k, deadline):
    """Pull all rows to their smallest valid positions."""
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    k_plus_1 = k + 1
    
    used_diffs = set()
    for row in best:
        used_diffs |= _row_to_diffs(row)
    
    for iteration in range(20):
        if time.time() > deadline - 5.0:
            break
        
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(1, k_plus_1):
                for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        for j in range(pos + 1, k_plus_1):
                            d = row[j] - new_val
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(row[pos] - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - row[pos])
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        any_changed = True
                        break
        
        if not any_changed:
            break
    
    return best


def _final_compact(solution, n_rows, k, deadline):
    """Final compaction pass with multiple restarts."""
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    for restart in range(5):
        if time.time() > deadline - 3.0:
            break
        
        # Perturb
        perturbed = [list(row) for row in best]
        for _ in range(5):
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k_plus_1 - 1)
            perturbed[row_idx][pos] += random.randint(1, 5)
        
        # Compact
        compacted = _pull_to_smallest_all(perturbed, n_rows, k, deadline)
        scope = max(max(row) for row in compacted)
        
        if scope < best_scope:
            best = compacted
            best_scope = scope
    
    return best
