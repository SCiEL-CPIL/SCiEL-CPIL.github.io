import time
import random
import math
import sys
import numpy as np
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1
    
    k_plus_1 = k + 1
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. SA-based row regeneration for coarse improvement
    if time.time() < deadline - 20.0:
        sa_solution, sa_scope = _sa_row_regeneration(best_solution, n_rows, k, deadline - 15.0)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_solution
            if best_scope <= lower_bound:
                return best_solution
    
    # 3. Fine-grained element-level local search for polish
    if time.time() < deadline - 12.0:
        polished, polish_scope = _element_local_search(best_solution, n_rows, k, deadline - 10.0)
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polished
            if best_scope <= lower_bound:
                return best_solution
        # Second pass of local search if time remains
        if time.time() < deadline - 8.0 and best_scope > lower_bound:
            polished2, polish_scope2 = _element_local_search(best_solution, n_rows, k, deadline - 6.0)
            if polish_scope2 < best_scope:
                best_scope = polish_scope2
                best_solution = polished2
                if best_scope <= lower_bound:
                    return best_solution
    
    # 4. Targeted backtracking search for further scope reduction
    if time.time() < deadline - 5.0:
        attempt = 0
        while time.time() < deadline - 5.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
            
            found = False
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                
                solution = _backtracking_search(n_rows, k, target_scope, deadline - 3.0, attempt)
                if solution:
                    new_scope = max(max(row) for row in solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = solution
                        found = True
                        # Polish the found solution
                        if time.time() < deadline - 3.0:
                            polished, ps = _element_local_search(best_solution, n_rows, k, deadline - 2.0)
                            if ps < best_scope:
                                best_scope = ps
                                best_solution = polished
                        if best_scope <= lower_bound:
                            return best_solution
                        break
                attempt += 1
            
            if not found and target_scope <= lower_bound + 5:
                break
    
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

def _sa_row_regeneration(solution, n_rows, k, deadline):
    """
    Simulated annealing with row regeneration.
    Coarse-grained moves: regenerate entire rows to escape local minima.
    """
    k_plus_1 = k + 1
    rows = [list(r) for r in solution]
    current_max = max(max(r) for r in rows)
    best_max = current_max
    best_rows = [list(r) for r in rows]
    
    # Track used differences
    used_diffs = set()
    for r in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(r[j] - r[i])
    
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    while time.time() < deadline:
        if T < T_min:
            T = 5.0
            cooling = 0.998
        
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                old_diffs.add(old_row[j] - old_row[i])
        
        used_diffs -= old_diffs
        
        # Regenerate row with randomization
        new_row = _generate_row_sa(used_diffs, k, current_max + 20)
        
        if new_row is None:
            used_diffs |= old_diffs
            continue
        
        new_row_max = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_row_max)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            new_diffs = set()
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    new_diffs.add(new_row[j] - new_row[i])
            used_diffs |= new_diffs
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
        else:
            used_diffs |= old_diffs
        
        T *= cooling
    
    return best_rows, best_max

def _generate_row_sa(used_diffs, k, max_search):
    """Generate a single row given the set of already-used differences."""
    k_plus_1 = k + 1
    row = [0]
    current_row_diffs = set()
    
    for _ in range(k):
        candidates = []
        x = row[-1] + 1
        forbidden = used_diffs | current_row_diffs
        
        while x < max_search and len(candidates) < 8:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden):
                candidates.append(x)
            x += 1
        
        if not candidates:
            return None
        
        # Biased random choice: prefer smaller values
        if random.random() < 0.7:
            next_x = candidates[0]
        else:
            next_x = random.choice(candidates)
        
        row.append(next_x)
        current_row_diffs |= {next_x - a for a in row[:-1]}
    
    return row

def _element_local_search(solution, n_rows, k, deadline):
    """
    Fine-grained element-level local search with conflict counting.
    Moves individual elements to reduce scope. Includes shifts and swaps for perturbation.
    """
    k_plus_1 = k + 1
    current_rows = [list(r) for r in solution]
    current_scope = max(max(r) for r in current_rows)
    best_scope = current_scope
    best_rows = [list(r) for r in current_rows]
    
    # Build diff counts for efficient conflict checking
    diff_counts = defaultdict(int)
    conflict_count = 0
    
    def add_diff(d):
        nonlocal conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1
    
    def remove_diff(d):
        nonlocal conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                conflict_count -= 1
    
    def row_diffs(row):
        """Get all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Initialize diff counts
    for r in current_rows:
        for d in row_diffs(r):
            add_diff(d)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    def try_replace(r_idx, p_idx, new_val, new_diffs):
        """Try replacing element at (r_idx, p_idx) with new_val and new_diffs."""
        nonlocal current_scope, best_scope, best_rows
        row = current_rows[r_idx]
        old_val = row[p_idx]
        
        # Remove old diffs
        diffs_to_remove = []
        for i in range(k_plus_1):
            if i != p_idx:
                d = abs(row[i] - old_val)
                diffs_to_remove.append(d)
        for d in diffs_to_remove:
            remove_diff(d)
        
        # Check if new_val is valid
        if new_val in row:
            for d in diffs_to_remove:
                add_diff(d)
            return False
        
        for a in row:
            if a == old_val:
                continue
            d = abs(new_val - a)
            if d in diff_counts and diff_counts[d] > 0:
                for dd in diffs_to_remove:
                    add_diff(dd)
                return False
        
        # Remove old, insert new
        row.pop(p_idx)
        insert_pos = 0
        while insert_pos < len(row) and row[insert_pos] < new_val:
            insert_pos += 1
        row.insert(insert_pos, new_val)
        
        for d in new_diffs:
            add_diff(d)
        
        # Check strictly increasing
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                row.pop(insert_pos)
                for d in new_diffs:
                    remove_diff(d)
                row.insert(p_idx, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                return False
        
        new_scope = max(max(r) for r in current_rows)
        delta = new_scope - current_scope
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
            return True
        else:
            row.pop(insert_pos)
            for d in new_diffs:
                remove_diff(d)
            row.insert(p_idx, old_val)
            for d in diffs_to_remove:
                add_diff(d)
            return False
    
    def try_shift(r_idx, p_idx, delta_val):
        """Try shifting element at (r_idx, p_idx) by delta_val."""
        nonlocal current_scope, best_scope, best_rows
        row = current_rows[r_idx]
        old_val = row[p_idx]
        new_val = old_val + delta_val
        
        if new_val <= 0:
            return False
        
        # Check bounds
        if p_idx > 0 and new_val <= row[p_idx - 1]:
            return False
        if p_idx < k_plus_1 - 1 and new_val >= row[p_idx + 1]:
            return False
        
        # Compute new diffs
        new_diffs = []
        for i in range(k_plus_1):
            if i != p_idx:
                new_diffs.append(abs(new_val - row[i]))
        
        return try_replace(r_idx, p_idx, new_val, new_diffs)
    
    def try_swap(r1_idx, p1_idx, r2_idx, p2_idx):
        """Try swapping elements at (r1_idx, p1_idx) and (r2_idx, p2_idx)."""
        nonlocal current_scope, best_scope, best_rows
        row1 = current_rows[r1_idx]
        row2 = current_rows[r2_idx]
        val1 = row1[p1_idx]
        val2 = row2[p2_idx]
        
        if val1 == val2:
            return False
        
        # Check if swap maintains strictly increasing
        if p1_idx > 0 and val2 <= row1[p1_idx - 1]:
            return False
        if p1_idx < k_plus_1 - 1 and val2 >= row1[p1_idx + 1]:
            return False
        if p2_idx > 0 and val1 <= row2[p2_idx - 1]:
            return False
        if p2_idx < k_plus_1 - 1 and val1 >= row2[p2_idx + 1]:
            return False
        
        # Remove all diffs involving both elements
        diffs1 = []
        for i in range(k_plus_1):
            if i != p1_idx:
                diffs1.append(abs(row1[i] - val1))
        diffs2 = []
        for i in range(k_plus_1):
            if i != p2_idx:
                diffs2.append(abs(row2[i] - val2))
        
        for d in diffs1:
            remove_diff(d)
        for d in diffs2:
            remove_diff(d)
        
        # Compute new diffs after swap
        new_diffs1 = []
        for i in range(k_plus_1):
            if i != p1_idx:
                new_diffs1.append(abs(row1[i] - val2))
        new_diffs2 = []
        for i in range(k_plus_1):
            if i != p2_idx:
                new_diffs2.append(abs(row2[i] - val1))
        
        # Check for conflicts
        for d in new_diffs1:
            if d in diff_counts and diff_counts[d] > 0:
                for dd in diffs1:
                    add_diff(dd)
                for dd in diffs2:
                    add_diff(dd)
                return False
        for d in new_diffs2:
            if d in diff_counts and diff_counts[d] > 0:
                for dd in diffs1:
                    add_diff(dd)
                for dd in diffs2:
                    add_diff(dd)
                return False
        
        # Perform swap
        row1[p1_idx] = val2
        row2[p2_idx] = val1
        
        for d in new_diffs1:
            add_diff(d)
        for d in new_diffs2:
            add_diff(d)
        
        new_scope = max(max(r) for r in current_rows)
        delta = new_scope - current_scope
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
            return True
        else:
            row1[p1_idx] = val1
            row2[p2_idx] = val2
            for d in new_diffs1:
                remove_diff(d)
            for d in new_diffs2:
                remove_diff(d)
            for d in diffs1:
                add_diff(d)
            for d in diffs2:
                add_diff(d)
            return False
    
    while time.time() < deadline:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
        
        # Choose move type: 60% replace, 25% shift, 15% swap
        move_type = random.random()
        r_idx = random.randint(0, n_rows - 1)
        p_idx = random.randint(1, k_plus_1 - 1)
        
        if move_type < 0.6:
            # Replace with random valid value
            row = current_rows[r_idx]
            old_val = row[p_idx]
            
            diffs_to_remove = []
            for i in range(k_plus_1):
                if i != p_idx:
                    d = abs(row[i] - old_val)
                    diffs_to_remove.append(d)
            for d in diffs_to_remove:
                remove_diff(d)
            
            max_search = min(best_scope + 10, 250)
            found = False
            new_val = None
            new_diffs = []
            
            for _ in range(20):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                
                valid = True
                new_diffs_candidate = []
                for a in row:
                    d = abs(cand - a)
                    if d in diff_counts and diff_counts[d] > 0:
                        valid = False
                        break
                    new_diffs_candidate.append(d)
                
                if valid:
                    new_val = cand
                    new_diffs = new_diffs_candidate
                    found = True
                    break
            
            if found:
                # Insert new value
                row.pop(p_idx)
                insert_pos = 0
                while insert_pos < len(row) and row[insert_pos] < new_val:
                    insert_pos += 1
                row.insert(insert_pos, new_val)
                
                for d in new_diffs:
                    add_diff(d)
                
                # Check strictly increasing
                ok = True
                for i in range(len(row) - 1):
                    if row[i] >= row[i + 1]:
                        ok = False
                        break
                
                if ok:
                    new_scope = max(max(r) for r in current_rows)
                    delta = new_scope - current_scope
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in current_rows]
                    else:
                        row.pop(insert_pos)
                        for d in new_diffs:
                            remove_diff(d)
                        row.insert(p_idx, old_val)
                        for d in diffs_to_remove:
                            add_diff(d)
                else:
                    row.pop(insert_pos)
                    for d in new_diffs:
                        remove_diff(d)
                    row.insert(p_idx, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
            else:
                for d in diffs_to_remove:
                    add_diff(d)
        
        elif move_type < 0.85:
            # Shift by small amount
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            try_shift(r_idx, p_idx, shift)
        
        else:
            # Swap with another row
            r2_idx = random.randint(0, n_rows - 1)
            p2_idx = random.randint(1, k_plus_1 - 1)
            if r2_idx != r_idx:
                try_swap(r_idx, p_idx, r2_idx, p2_idx)
        
        T *= cooling
    
    return best_rows, best_scope

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses faster conflict checking and better pruning.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_diffs = set()
    rows = []
    k_plus_1 = k + 1
    max_diff = max_scope  # Maximum possible difference
    
    def is_valid(x, row_so_far):
        """Check if x can be added to row_so_far without conflicts."""
        for a in row_so_far:
            if x - a in used_diffs:
                return False
        return True
    
    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Find valid candidates with tighter pruning
        valid_candidates = []
        remaining = k_plus_1 - pos - 1
        # Need room for 'remaining' more elements after x
        max_x = max_scope - remaining
        if max_x <= current_mark:
            return None
        
        # Also need at least 'remaining' values greater than x that are valid
        # Lower bound: x + remaining must be <= max_scope
        min_x = current_mark + 1
        max_x = min(max_scope - remaining, max_scope)
        
        if min_x > max_x:
            return None
        
        for x in range(min_x, max_x + 1):
            # Quick check: all differences x - a must not be in used_diffs
            valid = True
            for a in row_so_far:
                if x - a in used_diffs:
                    valid = False
                    break
            if valid:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Try candidates in order, biased towards smaller values
        n_to_try = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_to_try]
        
        # Occasionally try a larger candidate for diversity
        if len(valid_candidates) > n_to_try and rng.random() < 0.15:
            extra_idx = rng.randint(n_to_try, len(valid_candidates) - 1)
            candidates_to_try.append(valid_candidates[extra_idx])
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Compute new diffs
            new_diffs = [x - a for a in row_so_far]
            
            used_diffs.update(new_diffs)
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            used_diffs.difference_update(new_diffs)
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Add all diffs from this row
        for j in range(k_plus_1):
            for jj in range(j + 1, k_plus_1):
                used_diffs.add(row[jj] - row[j])
    
    return rows
