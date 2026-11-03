import time
import random
import sys
import math

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0, seed=None):
    """
    Greedy initialization with optional random skipping to create diversity.
    """
    if seed is not None:
        random.seed(seed)
    
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, skip to create diversity
                if skip_prob > 0 and random.random() < skip_prob:
                    # Skip a few valid values
                    skipped = False
                    for _ in range(random.randint(1, 3)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        # If we couldn't skip, just take current x if valid
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Simulated Annealing optimization.
    Cost = Scope + Penalty * Conflicts.
    """
    # Initial setup
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    
    # Estimate max possible scope to size the diff array
    # A safe upper bound for k=5, n=7 is around 200. 
    # We can dynamically expand if needed, but for speed we start with a generous bound.
    max_scope_guess = 250 
    # If initial rows have larger scope, expand
    init_scope = _get_scope(current_rows)
    if init_scope > max_scope_guess:
        max_scope_guess = init_scope + 50
        
    # used_diffs[d] = count of rows that have difference d
    used_diffs = [0] * (max_scope_guess + 1)
    
    # Initialize used_diffs
    for row in current_rows:
        diffs = _compute_diffs(row)
        for d in diffs:
            if d <= max_scope_guess:
                used_diffs[d] += 1
            else:
                # Should not happen if guess is good, but handle gracefully
                max_scope_guess = d + 10
                used_diffs.extend([0] * 10)
                used_diffs[d] += 1

    def get_cost():
        scope = _get_scope(current_rows)
        # Count conflicts: sum of (count - 1) for all d where count > 1
        # This is O(M) where M is max scope. 
        # To speed this up, we can maintain a running conflict count.
        # However, simple recalculation is O(250) which is very fast in Python if optimized? 
        # Actually, iterating 250 ints is fine.
        conflicts = 0
        for c in used_diffs:
            if c > 1:
                conflicts += c - 1
        # We want to minimize scope primarily, then conflicts.
        # Cost = scope + 10 * conflicts
        return scope + 10 * conflicts

    current_cost = get_cost()
    best_cost = current_cost
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.999 # Cooling rate
    
    while time.time() < deadline - 2:
        # Move selection
        # 1. Pick a row
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # 2. Pick a position to change (1 to k)
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # 3. Generate new value
        # Bias towards smaller values to reduce scope
        # But allow some random exploration
        if random.random() < 0.7:
            # Try to reduce or slightly vary
            lower = max(1, old_val - 10)
            upper = min(max_scope_guess, old_val + 10)
            new_val = random.randint(lower, upper)
        else:
            # Random jump
            new_val = random.randint(1, max_scope_guess)
            
        if new_val == old_val:
            continue
            
        # Check order constraints
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue
            
        # Update used_diffs
        # Remove old diffs involving pos
        old_diffs = []
        for i in range(pos):
            d = old_val - row[i]
            if d <= max_scope_guess:
                used_diffs[d] -= 1
                old_diffs.append(d)
        for j in range(pos + 1, len(row)):
            d = row[j] - old_val
            if d <= max_scope_guess:
                used_diffs[d] -= 1
                old_diffs.append(d)
                
        # Add new diffs involving pos
        new_diffs = []
        for i in range(pos):
            d = new_val - row[i]
            if d <= max_scope_guess:
                used_diffs[d] += 1
                new_diffs.append(d)
            else:
                # Expand array if necessary (rare)
                pass 
        for j in range(pos + 1, len(row)):
            d = row[j] - new_val
            if d <= max_scope_guess:
                used_diffs[d] += 1
                new_diffs.append(d)
            else:
                pass

        # Apply change
        row[pos] = new_val
        
        # Calculate new cost
        new_cost = get_cost()
        
        # Acceptance criterion
        if new_cost < current_cost or (T > 0 and random.random() < math.exp((current_cost - new_cost) / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            row[pos] = old_val
            for d in old_diffs:
                used_diffs[d] += 1
            for d in new_diffs:
                used_diffs[d] -= 1
                
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_rows

def _rebuild_row_greedy(forbidden, k, max_scope_hint):
    """
    Rebuild a row using greedy approach to minimize its maximum element.
    forbidden: set of differences already used by other rows.
    Returns the best row found or None.
    """
    row = [0]
    local_used = set()
    
    for _ in range(k):
        found = False
        # Try values from smallest possible
        last = row[-1]
        min_x = last + 1
        max_x = min(max_scope_hint, last + max_scope_hint)  # Upper bound
        
        for x in range(min_x, max_x + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & local_used:
                continue
            if new_diffs & forbidden:
                continue
            
            row.append(x)
            local_used |= new_diffs
            found = True
            break
        
        if not found:
            return None
    
    return row

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    
    # Multi-start strategy with different seeds and skip probabilities
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple initialization strategies with different seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3072]
    skip_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for seed, skip_prob in zip(seeds, skip_probs):
        if time.time() > deadline - 30:
            break
        
        rows, _ = _greedy_init(n_rows, k, skip_prob=skip_prob, seed=seed)
        if _is_valid(rows):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback to pure greedy
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, seed=42)
        return rows
    
    # Run SA optimization with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Use the best initial solution found
        rows = _sa_optimize(best_rows, n_rows, k, deadline)
        
        # Final check
        if _is_valid(rows):
            final_scope = _get_scope(rows)
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = [list(r) for r in rows]
    
    return best_rows
