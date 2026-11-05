# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses Simulated Annealing with bitset-based validity checks and a 
    targeted reduction polish phase.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We use 115s to be safe.
    deadline = start_time + 115.0

    def time_up():
        return time.time() > deadline

    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Fast Greedy Initialization
    def greedy_construct(seed, max_limit=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for pos in range(k):
                last = row[-1]
                found = False
                remaining = k - pos
                if max_limit is not None:
                    max_x = max_limit - (remaining - 1)
                else:
                    max_x = 100000
                    
                for x in range(last + 1, max_x + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        row.append(x)
                        local_used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    return None
            
            rows.append(row)
            used.update(local_used)
            
        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    
    # Try a few seeds to find a good initial solution
    for seed in range(20):
        if time_up(): break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        res = greedy_construct(0)
        if res:
            best_rows = res[0]
            best_max = get_max(best_rows)
        else:
            return []

    # 2. Simulated Annealing with Bitset Optimization
    # We maintain a boolean array `used_flags` for fast difference checking.
    # Size is dynamic based on current best_max + buffer.
    
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Determine initial size for used_flags
    # Max difference is at most current_max
    # We need size current_max + 1
    flag_size = current_max + 2 
    used_flags = [False] * flag_size
    
    # Populate used_flags
    for r in current_rows:
        for d in get_diffs(r):
            if d < flag_size:
                used_flags[d] = True
            else:
                # Should not happen if flag_size is correct, but safety
                flag_size = d + 2
                used_flags.extend([False] * (flag_size - len(used_flags)))
                used_flags[d] = True

    rng = random.Random(123)
    
    # Temperature schedule
    T_start = current_max * 0.5
    T_end = 1e-6
    T = T_start
    cooling_rate = 0.99995 # Slower cooling for more exploration
    
    # Helper to generate a random valid row given forbidden diffs (via flags)
    def generate_random_row(forbidden_flags, rng, max_val_limit, current_row_max_hint=None):
        row = [0]
        local_used = set()
        
        # We need to ensure local_used diffs don't conflict with global forbidden_flags
        # And don't conflict with each other.
        
        for pos in range(k):
            last = row[-1]
            remaining = k - pos
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            # Scan a window of candidates
            # To improve diversity, scan a larger window if possible
            # Limit window to avoid excessive computation
            scan_limit = min(max_x_allowed, last + 100)
            
            valid_candidates = []
            # We check candidates from last+1 to scan_limit
            # If we don't find enough, we might need to scan further, 
            # but for SA, a local window is usually sufficient for moves.
            
            for x in range(last + 1, scan_limit + 1):
                # Calculate new diffs
                new_diffs = []
                for a in row:
                    d = x - a
                    new_diffs.append(d)
                
                # Check validity
                # 1. Check against local_used
                # 2. Check against global forbidden_flags
                valid = True
                for d in new_diffs:
                    if d in local_used:
                        valid = False
                        break
                    if d < len(forbidden_flags) and forbidden_flags[d]:
                        valid = False
                        break
                    # If d >= len(forbidden_flags), it's automatically valid 
                    # because we only track up to max_val_limit which is <= len(forbidden_flags)-1
                    # Actually, forbidden_flags size is based on current_max.
                    # If max_val_limit is lower, d is smaller.
                    # If max_val_limit is higher (during restarts?), we might have issues.
                    # But we always constrain max_val_limit to <= current_max or best_max.
                    # So d <= current_max < len(forbidden_flags).
                
                if valid:
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 20:
                        break
            
            if not valid_candidates:
                # If local window fails, try a wider scan?
                # For speed, we just return None. SA will retry with different params.
                return None
            
            # Pick randomly
            x, new_diffs = rng.choice(valid_candidates)
            row.append(x)
            local_used.update(new_diffs)
            
        return row

    # SA Loop
    no_improve_steps = 0
    max_no_improve = 5000
    
    while not time_up():
        # Pick a random row to replace
        i = rng.randint(0, n_rows - 1)
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove old diffs from flags
        for d in old_diffs:
            if d < len(used_flags):
                used_flags[d] = False
        
        # Determine target max for new row
        if max(old_row) == current_max:
            # Try to improve
            target_max = current_max - 1
            if rng.random() < 0.1:
                target_max = current_max
        else:
            target_max = current_max
            if rng.random() < 0.05:
                 target_max = max(k, max(old_row) - 1)

        new_row = generate_random_row(used_flags, rng, target_max)
        
        if new_row is None:
            if target_max < current_max:
                new_row = generate_random_row(used_flags, rng, current_max)
            
            if new_row is None:
                # Restore old row
                for d in old_diffs:
                    if d < len(used_flags):
                        used_flags[d] = True
                no_improve_steps += 1
                continue
        
        new_diffs = get_diffs(new_row)
        
        # Double check validity (should be guaranteed)
        valid = True
        for d in new_diffs:
            if d < len(used_flags) and used_flags[d]:
                valid = False
                break
        if not valid:
            for d in old_diffs:
                if d < len(used_flags):
                    used_flags[d] = True
            no_improve_steps += 1
            continue
            
        # Calculate new global max
        current_rows[i] = new_row
        new_global_max = get_max(current_rows)
        
        delta = new_global_max - current_max
        
        if delta <= 0:
            # Improvement or equal
            for d in new_diffs:
                if d < len(used_flags):
                    used_flags[d] = True
            current_max = new_global_max
            no_improve_steps = 0
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Worse
            if T > 1e-9:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    # Accept
                    for d in new_diffs:
                        if d < len(used_flags):
                            used_flags[d] = True
                    current_max = new_global_max
                    no_improve_steps += 1
                else:
                    # Reject
                    current_rows[i] = old_row
                    for d in old_diffs:
                        if d < len(used_flags):
                            used_flags[d] = True
                    no_improve_steps += 1
            else:
                # Reject
                current_rows[i] = old_row
                for d in old_diffs:
                    if d < len(used_flags):
                        used_flags[d] = True
                no_improve_steps += 1
        
        # Cooling
        T *= cooling_rate
        
        # Optional: Restart if stuck
        if no_improve_steps > max_no_improve:
            # Try to restart with a slightly higher temperature or random perturbation
            T = max(T, current_max * 0.1)
            no_improve_steps = 0

    # 3. Final Polish: Targeted Reduction
    # Try to reduce the max by 1, 2, ... until we fail or time runs out.
    # This is a "high-precision polish pass".
    
    # Ensure best_rows is the current best
    if current_max < best_max:
        best_max = current_max
        best_rows = [r[:] for r in current_rows]
        
    # Re-sync flags for best_rows
    for d in range(len(used_flags)):
        used_flags[d] = False
    for r in best_rows:
        for d in get_diffs(r):
            if d < len(used_flags):
                used_flags[d] = True

    # Try to reduce best_max
    # We attempt to find a valid set of rows with max <= best_max - 1
    # If successful, update best_max and try again.
    
    # To speed up the polish, we use a more aggressive search.
    # We fix the "bottleneck" row (the one with max entry) and try to replace it.
    
    polish_time_limit = time.time() + 10.0 # Spend last 10s on polish
    
    while time.time() < polish_time_limit:
        target_limit = best_max - 1
        if target_limit < k:
            break
            
        # Identify rows that exceed target_limit
        bottleneck_indices = [i for i, r in enumerate(best_rows) if max(r) > target_limit]
        
        if not bottleneck_indices:
            # All rows are within limit, but we need to ensure global validity?
            # Actually, if we reduced best_max, we must have found valid rows.
            # If we just lowered the target, we need to find new rows.
            # Wait, the logic above:
            # We have best_rows with max = best_max.
            # We want to see if there exists a set of rows with max <= best_max - 1.
            # We can try to replace the rows that are currently > target_limit.
            
            # If no rows exceed target_limit, that means best_max was already <= target_limit?
            # No, best_max is the max of best_rows.
            # If max(r) <= target_limit for all r, then max(best_rows) <= target_limit < best_max.
            # This contradicts best_max being the max.
            # So there must be at least one row with max == best_max.
            
            pass
        
        # Try to replace one bottleneck row
        # Pick the one with the highest max
        bottleneck_indices.sort(key=lambda i: max(best_rows[i]), reverse=True)
        i = bottleneck_indices[0]
        
        old_row = best_rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove old diffs
        for d in old_diffs:
            if d < len(used_flags):
                used_flags[d] = False
        
        # Try to generate a new row with max <= target_limit
        # We use a more deterministic greedy approach with randomization for polish
        new_row = None
        for attempt in range(50):
            if time.time() > polish_time_limit:
                break
            # Randomize the order of candidate selection slightly
            # Or just use the same generate_random_row
            new_row = generate_random_row(used_flags, rng, target_limit)
            if new_row is not None:
                break
        
        if new_row is not None:
            # Check if this new row allows the rest of the rows to fit?
            # Since we only changed one row, and the other rows are <= best_max,
            # if we lower the target for this row, the global max becomes max(target_limit, max(other_rows)).
            # If max(other_rows) <= target_limit, then new global max is target_limit.
            # If max(other_rows) > target_limit, then global max is still max(other_rows).
            # But we want to reduce the GLOBAL max.
            # So we should only accept if max(other_rows) <= target_limit.
            
            other_max = max(max(r) for j, r in enumerate(best_rows) if j != i)
            
            if other_max <= target_limit:
                # Success!
                best_rows[i] = new_row
                for d in get_diffs(new_row):
                    if d < len(used_flags):
                        used_flags[d] = True
                best_max = target_limit
            else:
                # Fail, restore
                for d in old_diffs:
                    if d < len(used_flags):
                        used_flags[d] = True
        else:
            # Restore
            for d in old_diffs:
                if d < len(used_flags):
                    used_flags[d] = True

    # Final Validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True

    if not validate(best_rows, n_rows, k):
        # Fallback
        res = greedy_construct(0)
        if res:
            best_rows = res[0]
    
    return best_rows

# EVOLVE-BLOCK-END
