import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Greedy baseline to get an initial upper bound
    best_rows = _greedy_construct(n_rows, k, 0)
    best_scope = _scope(best_rows)
    
    # Phase 2a: Multiple greedy restarts with different random seeds
    for seed in range(1, 200):
        if time.time() > deadline - 30:
            break
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope == lower_bound:
                break
    
    # Phase 2b: Local search with shuffled row ordering to break symmetry
    if time.time() < deadline - 5:
        improved = _local_search_bitset(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 2c: Try to further improve with targeted row replacement
    if time.time() < deadline - 3:
        further = _targeted_improve(best_rows, n_rows, k, start_time, deadline)
        if further is not None:
            f_scope = _scope(further)
            if f_scope < best_scope:
                best_scope = f_scope
                best_rows = further
    
    # Phase 2d: Final polish pass - exhaustively try to shrink each row
    if time.time() < deadline - 1:
        polished = _polish_pass(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            p_scope = _scope(polished)
            if p_scope < best_scope:
                best_scope = p_scope
                best_rows = polished
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _greedy_construct(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 20 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 20:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            if not valid_xs:
                # Fallback: pick smallest valid
                chosen_x = last_x + 1
                while True:
                    new_diffs = {chosen_x - a for a in row}
                    if not (new_diffs & used):
                        break
                    chosen_x += 1
            else:
                chosen_x = rng.choice(valid_xs)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _local_search_bitset(initial_rows, n_rows, k, start_time, deadline):
    """Local search using bitset-based conflict detection for speed with shuffled ordering."""
    rng = random.Random(12345)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff sets for all rows
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    max_steps = 50000
    T = 20.0
    cool_rate = 0.9995
    
    # Create shuffled order to break symmetry
    order = list(range(n_rows))
    rng.shuffle(order)
    step_in_order = 0
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Use shuffled ordering to break symmetry-dependent local minima
        idx = order[step_in_order % n_rows]
        step_in_order += 1
        
        old_row = list(rows[idx])
        old_diffs = row_diff_sets[idx]
        
        # Generate a perturbed row: try to replace with a slightly different one
        new_row = _perturb_row(old_row, rows, idx, k, rng)
        
        if new_row is None:
            continue
        
        new_diffs = _row_diffs(new_row)
        
        # Check if new_diffs conflicts with any other row's diffs
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_diffs & row_diff_sets[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Valid move: check scope
        new_rows = rows[:]
        new_rows[idx] = new_row
        new_scope = _scope(new_rows)
        
        delta = new_scope - current_scope
        if delta <= 0:
            # Accept improvement
            rows = new_rows
            row_diff_sets[idx] = new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Metropolis acceptance
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows = new_rows
                row_diff_sets[idx] = new_diffs
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows

def _perturb_row(old_row, all_rows, exclude_idx, k, rng):
    """Generate a perturbed version of a row that may have lower scope."""
    # Strategy: try to shift some elements down or up slightly
    row = list(old_row)
    n = len(row)
    
    # Try a few different perturbation strategies
    for _ in range(5):
        candidate = list(old_row)
        
        # Randomly pick 1-3 positions to modify
        num_mods = rng.randint(1, 3)
        positions = rng.sample(range(1, n), min(num_mods, n - 1))
        
        for pos in positions:
            # Try to adjust this element
            old_val = candidate[pos]
            # Try values nearby
            for delta_val in [rng.randint(-3, 3)]:
                new_val = old_val + delta_val
                if new_val <= 0:
                    continue
                # Check strictly increasing
                if pos > 0 and new_val <= candidate[pos - 1]:
                    continue
                if pos < n - 1 and new_val >= candidate[pos + 1]:
                    continue
                
                # Check if this creates internal conflicts
                # (differences within the row must be distinct)
                candidate[pos] = new_val
                diffs = set()
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        d = candidate[j] - candidate[i]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                if not valid:
                    candidate[pos] = old_val
                    continue
                break  # Keep this modification if valid
        
        # Verify the entire row is valid (strictly increasing, distinct internal diffs)
        if not _is_valid_row(candidate):
            continue
        
        return candidate
    
    return None

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _targeted_improve(initial_rows, n_rows, k, start_time, deadline):
    """Try to improve by replacing individual rows with better alternatives."""
    rng = random.Random(99999)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(1000):
        if time.time() > deadline - 3:
            break
        
        # Pick a random row
        idx = rng.randint(0, n_rows - 1)
        
        # Collect diffs from all other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            rd = _row_diffs(rows[j])
            used_diffs.update(rd)
        
        # Try to find a row with smaller max value that avoids used_diffs
        old_max = rows[idx][-1]
        
        # Binary search for minimum possible max
        lo = k  # minimum possible max (0,1,2,...,k)
        hi = old_max
        
        best_new_row = None
        for _ in range(20):  # Try a few random constructions
            # Try to construct a row with max <= hi
            new_row = _construct_row_with_limit(used_diffs, k, hi, rng)
            if new_row is not None:
                if best_new_row is None or new_row[-1] < best_new_row[-1]:
                    best_new_row = new_row
                if best_new_row[-1] == lo:
                    break
        
        if best_new_row is not None and best_new_row[-1] < old_max:
            rows[idx] = best_new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        
        # Also try random perturbations
        if time.time() > deadline - 3:
            break
        perturbed = _perturb_row(rows[idx], rows, idx, k, rng)
        if perturbed is not None:
            # Check conflicts
            new_diffs = _row_diffs(perturbed)
            conflict = False
            for j in range(n_rows):
                if j == idx:
                    continue
                if new_diffs & _row_diffs(rows[j]):
                    conflict = True
                    break
            if not conflict:
                rows[idx] = perturbed
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
    
    return best_rows

def _polish_pass(initial_rows, n_rows, k, start_time, deadline):
    """Final polish pass: exhaustively try to shrink each row's maximum element."""
    rng = random.Random(77777)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # For each row, try to reduce its maximum by finding an alternative
    for idx in range(n_rows):
        if time.time() > deadline - 1:
            break
        
        old_row = rows[idx]
        old_max = old_row[-1]
        
        # Collect diffs from all other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            rd = _row_diffs(rows[j])
            used_diffs.update(rd)
        
        # Try to find a row with max < old_max that avoids used_diffs
        # Start from old_max - 1 and go down
        for target_max in range(old_max - 1, k - 1, -1):
            if time.time() > deadline - 1:
                break
            
            # Try multiple random constructions with this limit
            found = False
            for attempt in range(50):
                new_row = _construct_row_with_limit(used_diffs, k, target_max, rng)
                if new_row is not None:
                    rows[idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    found = True
                    break
            
            if found:
                break
        
        # Also try to reduce individual elements (not just the max)
        if time.time() > deadline - 1:
            break
        for pos in range(k, 0, -1):  # Try to reduce from the end
            if time.time() > deadline - 1:
                break
            
            old_val = rows[idx][pos]
            # Try to reduce this element
            for new_val in range(old_val - 1, rows[idx][pos-1], -1):
                candidate = list(rows[idx])
                candidate[pos] = new_val
                
                # Check validity
                if not _is_valid_row(candidate):
                    continue
                
                new_diffs = _row_diffs(candidate)
                
                # Check conflicts with other rows
                conflict = False
                for j in range(n_rows):
                    if j == idx:
                        continue
                    if new_diffs & _row_diffs(rows[j]):
                        conflict = True
                        break
                
                if not conflict:
                    rows[idx] = candidate
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break  # Stop reducing this position
    
    return best_rows

def _construct_row_with_limit(used_diffs, k, max_val, rng):
    """Construct a row of k+1 elements starting at 0, max <= max_val,
    with all internal diffs avoiding used_diffs."""
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        
        x = last_x + 1
        while x <= max_val:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        # Choose randomly from valid options, biased toward smaller
        if rng.random() < 0.5:
            chosen = valid_xs[0]  # Smallest
        else:
            chosen = rng.choice(valid_xs[:min(10, len(valid_xs))])
        
        row.append(chosen)
        for a in row[:-1]:
            used_diffs.add(chosen - a)
    
    return row
