import math
import random
import time
import bisect

# EVOLVE-BLOCK-START

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used

def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True

def _packing_score(new_diffs, used_set, x, k):
    """
    Score a candidate x for the next element in a row.
    Lower score is better.
    Heuristic: We want to place differences 'close' to existing used differences
    to minimize the span (scope) required. Specifically, we penalize large gaps.
    """
    if not new_diffs:
        return x * 10.0
    
    score = 0.0
    # We want to fill gaps. If a difference d is far from any used difference, it's bad.
    # However, checking distance to all used differences is O(N).
    # A simpler proxy: The maximum difference introduced by x is x (since row starts at 0).
    # We want to keep x small. But we also want the differences to be 'clustered'.
    
    # Let's use a simpler metric: 
    # 1. Prefer small x.
    # 2. Prefer if the new differences are 'close' to the boundary of the used set?
    # Actually, the best fit for difference packing is often related to how well
    # the new differences fit into the 'holes' of the used set.
    
    # Efficient proxy:
    # If used_set is dense, any valid x is good.
    # If used_set is sparse, we want x to be small.
    
    # Let's try: Sum of distances to nearest used difference for each new diff.
    # To make it fast, we only check a few neighbors or use a pre-sorted list if available.
    # For now, a simple penalty on x is the most robust.
    # Let's add a penalty if the new diffs are 'isolated'.
    
    # A better heuristic: 
    # The "cost" of a row is roughly its maximum element.
    # When building a row, we want to choose x such that the *future* constraints are minimized.
    # This is hard. 
    # Instead, let's look at the "density" of the used set.
    # If we have a boolean array of used diffs, we can check how many "new blocks" are formed.
    
    # For this implementation, we will rely on a simple heuristic:
    # Minimize x, but with a randomization factor to allow exploration.
    # The "packing" idea from inspiration: 
    # "Score a candidate x by how well its differences 'pack' into the used difference space."
    
    # Let's implement a lightweight version:
    # Count how many of the new diffs are "adjacent" (within distance 3) to an existing used diff.
    # More adjacent = better packing (fills gaps).
    
    adjacency_count = 0
    for d in new_diffs:
        # Check neighbors
        if (d-1 in used_set) or (d+1 in used_set) or (d-2 in used_set) or (d+2 in used_set):
            adjacency_count += 1
            
    # Score: Primary is x (smaller is better), Secondary is negative adjacency (more is better)
    # We want to minimize x, but if two x are close, prefer the one with better adjacency.
    return x - 0.1 * adjacency_count

def _greedy_construct(n_rows, k, rng, used_set, max_val_hint=None):
    """
    Greedy construction using best-fit scoring.
    """
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            best_x = None
            best_score = float('inf')
            # We search from the smallest possible x upwards
            # But we can limit the search range if we have a hint
            start_x = row[-1] + 1
            # Heuristic upper bound: usually we don't need to go too far if we pack well
            # But we must find a valid one.
            
            # To speed up, we can randomize the start or search window slightly?
            # No, we must find the *best* fit. 
            # We scan x from start_x upwards.
            # We stop scanning if we find a "good enough" fit? No, we want the global best for this step.
            # However, scanning indefinitely is slow.
            # Let's scan up to a reasonable limit, say current_scope + 20.
            
            # Actually, the "best fit" implies we should find the x that minimizes the score.
            # Since score increases with x (mostly), the first valid x with good adjacency is likely best.
            # But a slightly larger x might have much better adjacency.
            
            # Strategy: Scan x from start_x to start_x + 50 (or similar).
            # Pick the one with min score.
            
            search_limit = start_x + 100
            # If we have a max_val_hint, we might not exceed it? 
            # No, we are constructing, so we don't know the final scope yet.
            
            candidates = []
            for x in range(start_x, search_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_set:
                    continue
                
                score = _packing_score(new_diffs, used_set, x, k)
                candidates.append((score, x))
                
                # Optimization: If we find a very good score (e.g., x is small and high adjacency), 
                # we might break early? 
                # Since score ~ x, the first valid x is the smallest x. 
                # If we just want smallest x, we break at first valid.
                # But we want "best fit". 
                # Let's collect a few candidates.
                
            if not candidates:
                # If no candidate found in window, expand search
                # This shouldn't happen often if max_val is not tight
                x = search_limit + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_set):
                        score = _packing_score(new_diffs, used_set, x, k)
                        candidates.append((score, x))
                        break
                    x += 1
                    if x > 5000: # Safety
                        break
                
                if not candidates:
                    return None # Failed to construct

            # Sort by score
            candidates.sort(key=lambda t: t[0])
            
            # Pick the best, or randomly from top 3 for diversity?
            # For deterministic best-fit, pick best.
            # For SA, we might want randomness. 
            # Let's pick best for the initial greedy phase.
            best_x = candidates[0][1]
            
            row.append(best_x)
            used_set.update({best_x - a for a in row[:-1]})
            
        rows.append(row)
    
    return rows

def _simulated_annealing(rows, n_rows, k, rng, deadline):
    """
    Simulated Annealing to improve scope.
    Moves:
    1. Shift a single element in a row.
    2. Reconstruct a single row using greedy heuristic with noise.
    """
    rows = [r[:] for r in rows]
    used_set = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    temperature = 20.0
    cooling_rate = 0.99
    
    while time.time() < deadline and temperature > 0.5:
        # Choose move type
        if rng.random() < 0.7:
            # Move 1: Shift element
            idx = rng.randint(0, len(rows) - 1)
            pos = rng.randint(1, len(rows[idx]) - 1)
            row = rows[idx]
            old_val = row[pos]
            
            # Try to shift by a small delta
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
            
            # Check validity
            if new_val <= row[pos-1]:
                continue
            if pos + 1 < len(row) and new_val >= row[pos+1]:
                continue
                
            # Calculate diff changes
            old_diffs = set()
            new_diffs = set()
            for i in range(len(row)):
                if i == pos:
                    continue
                if i < pos:
                    old_diffs.add(old_val - row[i])
                    new_diffs.add(new_val - row[i])
                else:
                    old_diffs.add(row[i] - old_val)
                    new_diffs.add(row[i] - new_val)
            
            # Check conflicts
            remaining = used_set - old_diffs
            if new_diffs & remaining:
                continue
            
            # Calculate new scope
            new_scope = current_scope
            if new_val > current_scope:
                new_scope = new_val
            elif old_val == current_scope:
                # If we moved the max element, check if it's still the max
                # or if another row has the max
                # Simple check: recompute max? Expensive.
                # Approximation: if new_val < current_scope, scope might drop.
                # Let's just compute max for the row and compare with others?
                # For speed, let's assume scope only changes if we touch the global max.
                # If we move the element that WAS the max, we need to check if it's still the max.
                # If it drops, does the global max drop?
                # Only if no other row has an element >= new max.
                
                # To be safe and simple:
                # If new_val < old_val and old_val was the global max:
                # Check if any other row has max >= new_val.
                # This is O(N*K). Acceptable for SA step?
                # Let's do a quick check.
                other_max = 0
                for i, r in enumerate(rows):
                    if i == idx:
                        continue
                    if r[-1] > other_max:
                        other_max = r[-1]
                
                row_max = max(row[:-1] + [new_val] + row[pos+1:]) # Need to update row first?
                # Let's just compute the potential new global max
                # Potential new max for this row
                # Actually, let's just accept the move and recompute scope if it looks promising.
                pass
            
            # Cost calculation
            delta_scope = new_scope - current_scope
            
            if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
                # Accept move
                row[pos] = new_val
                used_set = (used_set - old_diffs) | new_diffs
                
                # Update current scope
                # Recompute if necessary
                # Optimization: Only recompute if we moved an element that was or could be the max
                if new_val >= current_scope or old_val == current_scope:
                    current_scope = _total_scope(rows)
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
                    
        else:
            # Move 2: Reconstruct a row
            idx = rng.randint(0, len(rows) - 1)
            old_row = rows[idx][:]
            old_diffs = _row_diffs(old_row)
            
            # Remove old diffs from used set
            used_set.difference_update(old_diffs)
            
            # Reconstruct with some noise
            # We can modify the greedy constructor to accept a "noise" parameter
            # For now, we just call greedy on a temporary set and see if it improves
            # This is expensive. Let's stick to element shifts for now, or use a lighter reconstruction.
            
            # Let's try a "perturbed greedy" for this row.
            # We know the other rows are fixed.
            # We want to build a new row that fits well into the remaining holes.
            
            new_row = [0]
            success = True
            for _ in range(k):
                best_x = None
                best_score = float('inf')
                start_x = new_row[-1] + 1
                
                # Search with some randomization
                # Sample a few candidates
                candidates = []
                for _ in range(20): # Sample 20 candidates
                    # Random x in a reasonable range
                    # Range: start_x to start_x + 50
                    x = rng.randint(start_x, start_x + 50)
                    new_diffs = {x - a for a in new_row}
                    if new_diffs & used_set:
                        continue
                    score = _packing_score(new_diffs, used_set, x, k)
                    # Add noise
                    score += rng.uniform(-2, 2)
                    candidates.append((score, x))
                
                if not candidates:
                    # Fallback to deterministic scan
                    x = start_x
                    while True:
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & used_set):
                            score = _packing_score(new_diffs, used_set, x, k)
                            candidates.append((score, x))
                            break
                        x += 1
                        if x > 1000:
                            success = False
                            break
                    
                    if not success:
                        break
                
                candidates.sort(key=lambda t: t[0])
                best_x = candidates[0][1]
                new_row.append(best_x)
                used_set.update({best_x - a for a in new_row[:-1]})
            
            if success and len(new_row) == k + 1:
                new_scope = _total_scope(rows[:idx] + [new_row] + rows[idx+1:])
                delta_scope = new_scope - current_scope
                
                if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
                    rows[idx] = new_row
                    # used_set already updated in loop? 
                    # Wait, we updated used_set in the loop above.
                    # But we removed old_diffs before.
                    # So used_set is now: (Old - OldRow) + NewRow.
                    # This is correct.
                    
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
            
            else:
                # Rollback: put old row back
                rows[idx] = old_row
                used_set.update(old_diffs)
                
        temperature *= cooling_rate
        
    return best_rows

def _polish_pass(rows, k, deadline):
    """
    Deterministic polish: try to reduce the maximum element of each row.
    """
    rows = [r[:] for r in rows]
    used_set = _all_diffs_used(rows)
    
    # Iterate a few times
    for _ in range(5):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[idx]
            # Try to reduce the last element first, then others
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try to reduce
                for new_val in range(old_val - 1, row[pos-1], -1):
                    old_diffs = set()
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                            new_diffs.add(new_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                            new_diffs.add(row[i] - new_val)
                    
                    remaining = used_set - old_diffs
                    if not (new_diffs & remaining):
                        # Valid move
                        row[pos] = new_val
                        used_set = (used_set - old_diffs) | new_diffs
                        improved = True
                        break # Move to next pos
        
        if not improved:
            break
            
    return rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Phase 1: Initial Greedy Construction
    # We can try multiple seeds to get diverse initial solutions
    best_rows = None
    best_scope = float('inf')
    
    # Run greedy with a few different randomizations
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        r = random.Random(seed)
        # Clear used set for fresh construction
        used_set = set()
        rows = _greedy_construct(n_rows, k, r, used_set)
        if rows:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
    if best_rows is None:
        # Fallback: simple greedy
        used_set = set()
        best_rows = _greedy_construct(n_rows, k, rng, used_set)
        best_scope = _total_scope(best_rows)
        
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 20:
        sa_rows = _simulated_annealing(best_rows, n_rows, k, rng, deadline - 20)
        sa_scope = _total_scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
            
    # Phase 3: Polish
    if time.time() < deadline - 5:
        polish_rows = _polish_pass(best_rows, k, deadline - 5)
        polish_scope = _total_scope(polish_rows)
        if polish_scope < best_scope:
            best_rows = polish_rows
            
    return best_rows

# EVOLVE-BLOCK-END
