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
    
    # Phase 2: Try backtracking with forward checking to find a better solution
    # We try scopes from the lower bound up to the current best scope
    # If we find a valid set of rows for scope M, we update best_scope
    # and can stop if we hit the lower bound
    
    # To avoid spending too much time, we limit the backtracking attempts
    # We will try to solve for M in [lower_bound, best_scope - 1]
    # If time is running out, we break and return best_rows
    
    # We use a randomized restart strategy for backtracking to avoid getting stuck
    # in a specific ordering of rows
    rng = random.Random(42)
    
    # Try a few different max_scope values, starting from the best known
    # and going down. Or, better, try to find the *minimum* possible.
    # Since we have a deadline, we can't do a full binary search if each step is expensive.
    # However, for n=7, k=6, the search space is small enough that a well-pruned
    # backtracker might find the optimal or near-optimal quickly.
    
    # Let's try to improve upon the greedy result.
    # We will attempt to construct a solution with max_entry < best_scope.
    # We iterate potential max_entries downwards from best_scope - 1.
    
    # Optimization: The backtracker needs to be fast.
    # It will use a set of used differences.
    
    # We will try a few random orderings of rows to break symmetry.
    # For each ordering, we try to build the rows sequentially.
    
    # To save time, we will only run this if we have enough time left.
    if time.time() < deadline - 10:
        # Try to find a better solution using backtracking
        # We'll try a range of max_scopes.
        # Given the problem size, we can afford a few seconds of backtracking.
        
        # Strategy: 
        # 1. Get current best_scope.
        # 2. Try to find a solution with max_scope = best_scope - 1.
        # 3. If found, update best_scope and try best_scope - 2, etc.
        # 4. Stop if time runs out or we hit lower_bound.
        
        current_best_scope = best_scope
        
        # We will try a limited number of backtracking attempts per scope
        # to avoid getting stuck in a deep, unproductive branch.
        max_backtrack_attempts = 50
        
        for target_scope in range(current_best_scope - 1, lower_bound - 1, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            for attempt in range(max_backtrack_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Shuffle row indices to break symmetry
                row_order = list(range(n_rows))
                rng.shuffle(row_order)
                
                # Run backtracking with this specific row order and target scope
                solution = _backtrack_solve(n_rows, k, target_scope, row_order, deadline)
                
                if solution is not None:
                    # Check validity just in case
                    if _is_valid_solution(solution, n_rows, k):
                        new_scope = _scope(solution)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in solution]
                        found = True
                        break # If we found one for this scope, we don't need more for this scope
                        # Actually, we want the MINIMUM scope. If we found one for target_scope,
                        # we can try target_scope - 1 next.
            
            if not found:
                # If we failed to find a solution for target_scope, 
                # it's likely that target_scope - 1 will also fail or be harder.
                # But due to randomness, maybe we just got unlucky.
                # However, if we can't find ANY solution for a scope, 
                # it's safer to assume the next lower scope is also impossible 
                # unless we try many more times. 
                # For efficiency, if we fail for a scope, we might skip lower scopes 
                # unless we have lots of time. 
                # Let's just break if we fail for the first target_scope (best_scope - 1) 
                # to save time for other methods? 
                # No, let's try a couple more.
                pass
    
    # Phase 3: Local search to polish the result
    if time.time() < deadline - 5:
        improved = _local_search_bitset(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

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

def _is_valid_solution(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i+1] <= row[i]:
                return False
        diffs = _row_diffs(row)
        if len(diffs) != (k+1)*k//2:
            return False
        if all_diffs & diffs:
            return False
        all_diffs.update(diffs)
    return True

def _backtrack_solve(n_rows, k, max_scope, row_order, deadline):
    """
    Attempts to construct n_rows rows with max entry <= max_scope.
    Uses backtracking with forward checking.
    row_order: list of indices indicating the order in which to construct rows.
    """
    used_diffs = set()
    rows = [None] * n_rows
    
    # We need to pick rows one by one.
    # The constraint is that all differences must be unique across all rows.
    # Total differences needed: n_rows * (k+1)*k//2
    # Available differences: 1 to max_scope
    # If max_scope < total_diffs_needed, it's impossible.
    total_needed = n_rows * (k+1)*k//2
    if max_scope < total_needed:
        return None
        
    def backtrack(idx):
        if time.time() > deadline - 2:
            return False
        
        if idx == n_rows:
            return True
            
        row_idx = row_order[idx]
        
        # Generate candidate rows for this position
        # We need to find a row of k+1 integers starting at 0, max <= max_scope
        # such that its differences do not intersect with used_diffs.
        
        # We use a greedy-with-backtracking approach to generate the row.
        # Since k is small (6), we can backtrack within the row construction too.
        
        def build_row(pos, current_row, row_used_diffs):
            if time.time() > deadline - 2:
                return False
            
            if pos == k + 1:
                # Check if this row is valid (it should be by construction)
                # and if it conflicts with global used_diffs (it shouldn't)
                # Update global state
                used_diffs.update(row_used_diffs)
                rows[row_idx] = current_row
                
                # Recurse to next row
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for d in row_used_diffs:
                    used_diffs.discard(d)
                rows[row_idx] = None
                return False
            
            last_val = current_row[-1]
            
            # Try next values for current_row[pos]
            # Start from last_val + 1 up to max_scope
            # But we must ensure we have enough "room" to place the rest of the row
            # and that the differences don't conflict.
            
            # Heuristic: Try smaller values first to keep the row compact?
            # Or try to leave more room?
            # Let's try values in increasing order.
            
            start_val = last_val + 1
            # Optimization: We must be able to fit (k+1 - pos - 1) more elements.
            # So the current element cannot be too large.
            # Max value for this element: max_scope - (k+1 - pos - 1)
            # Actually, strictly increasing integers, so min gap is 1.
            # So val <= max_scope - (k + 1 - pos - 1)
            
            end_val = max_scope - (k + 1 - pos - 1)
            
            if start_val > end_val:
                return False
                
            for val in range(start_val, end_val + 1):
                # Check differences with previous elements in this row
                new_diffs = set()
                valid = True
                for prev in current_row:
                    d = val - prev
                    if d in row_used_diffs:
                        valid = False
                        break
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                    
                # Forward checking: 
                # If we pick this val, do we still have enough differences available
                # for the rest of this row and the remaining rows?
                
                # This is a complex check. A simpler check:
                # Is there at least one valid completion for the rest of this row?
                # We can approximate by checking if the number of available diffs
                # is sufficient.
                
                # Let's just recurse.
                
                current_row.append(val)
                row_used_diffs.update(new_diffs)
                
                if build_row(pos + 1, current_row, row_used_diffs):
                    return True
                    
                current_row.pop()
                for d in new_diffs:
                    row_used_diffs.discard(d)
                    
            return False
            
        # Start building the row
        if build_row(1, [0], set()):
            return True
            
        return False

    if backtrack(0):
        return rows
    return None
