import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Determine a reasonable upper bound for the max entry to size the histogram
    # Greedy initial estimate
    def greedy_estimate():
        used = set()
        max_val = 0
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            max_val = max(max_val, max(row))
        return max_val

    initial_max = greedy_estimate()
    # Add some buffer
    max_bound = int(initial_max * 1.2) + 50
    if max_bound < 100:
        max_bound = 100

    # diff_count[d] = number of times difference d is used across all rows
    diff_count = [0] * (max_bound + 1)

    def update_diff_counts(row, delta):
        """
        Update diff_count for a row. delta=1 to add, delta=-1 to remove.
        """
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= max_bound:
                    diff_count[d] += delta
                else:
                    # This shouldn't happen if we respect bounds, but safety
                    pass

    def get_row_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

    def build_row_with_gaps(max_val_limit, collision_weight=1.0, value_weight=1.0):
        """
        Builds a row by selecting gaps g_1, ..., g_k.
        Uses global diff_count to estimate collision cost.
        """
        row = [0]
        # We will pick k gaps
        for step in range(k):
            last_val = row[-1]
            best_gap = None
            best_cost = float('inf')
            best_new_diffs = None
            
            # Search for gap g such that last_val + g <= max_val_limit
            # We look for g in a reasonable range.
            # To keep max small, we prefer small g, but collision cost matters.
            # We scan g from 1 upwards until we find a good candidate or reach limit.
            # Since k is small, we can be a bit exhaustive on small g.
            
            # Heuristic: check g in [1, 20] first, then wider if needed?
            # Actually, just scan 1 to max_val_limit - last_val.
            # But that's too slow if max_val_limit is large.
            # Let's scan 1 to min(50, max_val_limit - last_val) first.
            # If no good candidate, expand.
            
            search_limit = min(50, max_val_limit - last_val)
            
            for g in range(1, search_limit + 1):
                new_val = last_val + g
                if new_val > max_val_limit:
                    break
                
                # Calculate new differences
                new_diffs = [new_val - a for a in row]
                
                # Check if any diff is already in this row (local conflict)
                # The row must have distinct differences internally?
                # The problem says "all differences between two entries of the same row are distinct over all rows".
                # This implies differences are distinct globally, which implies they are distinct locally.
                # So we must check local distinctness too.
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Check global collision cost
                cost = 0
                for d in new_diffs:
                    if d <= max_bound:
                        cost += diff_count[d]
                    else:
                        cost += 1000 # Penalize out of bounds heavily
                
                # Combine cost with value penalty
                # We want to minimize max entry, so penalize large new_val or large g
                total_cost = collision_weight * cost + value_weight * new_val
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_gap = g
                    best_new_diffs = new_diffs
            
            if best_gap is None:
                # Fallback: try larger gaps if we didn't find one in the small range
                # This can happen if small gaps cause too many collisions
                for g in range(search_limit + 1, max_val_limit - last_val + 1):
                    new_val = last_val + g
                    if new_val > max_val_limit:
                        break
                    
                    new_diffs = [new_val - a for a in row]
                    if len(set(new_diffs)) != len(new_diffs):
                        continue
                    
                    cost = 0
                    for d in new_diffs:
                        if d <= max_bound:
                            cost += diff_count[d]
                        else:
                            cost += 1000
                    
                    total_cost = collision_weight * cost + value_weight * new_val
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_gap = g
                        best_new_diffs = new_diffs
            
            if best_gap is None:
                return None
            
            row.append(last_val + best_gap)
            
            # Note: We do NOT update diff_count yet, as the row is not finalized.
            # We only use the current global state for lookahead.
            # This is a greedy approximation.
            
        return row

    # Initial construction
    rows = []
    # Sort rows by some heuristic? Maybe build the "hardest" ones first?
    # Or just build sequentially.
    for i in range(n_rows):
        # Try to build with a tight limit first
        # Estimate remaining capacity
        current_max = max((max(r) for r in rows), default=0)
        # We don't know the final max yet. Let's just use max_bound.
        # Actually, we can try to keep max small.
        # Let's use a dynamic limit: current_max + (initial_max - current_max) / (n_rows - i) ?
        # Simpler: just use max_bound.
        
        row = build_row_with_gaps(max_bound, collision_weight=1.0, value_weight=0.1)
        if row is None:
            # Fallback to simple greedy
            row = [0]
            used_local = set()
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_local):
                        break
                    x += 1
                row.append(x)
                used_local.update(new_diffs)
        
        rows.append(row)
        update_diff_counts(row, 1)
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Local Search / Optimization
    rng = random.Random(42)
    temp = 1.0
    cooling = 0.99
    
    # We want to reduce the max entry.
    # Strategy: Pick a row, remove it, try to rebuild it with a smaller max limit.
    # If successful, accept.
    
    current_max = best_max
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
            
        # Pick a random row to rebuild
        i = rng.randint(0, n_rows - 1)
        old_row = rows[i]
        old_max = max(old_row)
        
        # Remove from global counts
        update_diff_counts(old_row, -1)
        
        # Determine target max
        # We want to reduce the global max.
        # If this row contains the current global max, we must reduce its max.
        # Otherwise, we can try to keep it small to free up space for others?
        # Actually, reducing any row's max might allow others to be smaller?
        # No, the constraint is global distinctness.
        # If we make a row smaller, it uses smaller differences, which might be more crowded.
        # But it might allow the *global* max to be lower if this row was the bottleneck.
        
        # Let's try to rebuild with a max limit slightly smaller than the current row max,
        # or smaller than the global max if this row is the max holder.
        
        if old_max == current_max:
            target_max = current_max - 1
        else:
            # Try to keep it compact, maybe slightly smaller than its current max
            target_max = old_max - 1 if old_max > k else old_max
            
        # Ensure target_max is at least k
        if target_max < k:
            target_max = k
            
        # Try to rebuild
        # Use higher collision weight to avoid crowded spots
        new_row = build_row_with_gaps(target_max, collision_weight=2.0, value_weight=0.2)
        
        if new_row is None:
            # Try with slightly larger limit
            new_row = build_row_with_gaps(old_max, collision_weight=2.0, value_weight=0.2)
            
        if new_row is None:
            # Restore
            update_diff_counts(old_row, 1)
            continue
        
        new_max = max(new_row)
        
        # Check if this improves the global max
        # Calculate new global max if we replace row i
        # We can just check if new_max < old_max and old_max == current_max
        # Or more generally, if the max of all rows becomes smaller.
        
        # Efficient check:
        # If old_max was the unique global max, and new_max < old_max, then global max decreases.
        # If old_max was not the global max, global max stays same (unless new_max is larger, which we avoid).
        
        # Let's calculate the actual new global max to be safe, but it's O(n_rows).
        # Since n_rows is small, it's fine.
        
        # Temporarily replace
        rows[i] = new_row
        new_global_max = max(max(r) for r in rows)
        
        if new_global_max < current_max:
            # Accept
            update_diff_counts(new_row, 1)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
            best_max = current_max
            temp = max(temp * 1.1, 0.1) # Reheat
        else:
            # Neutral or worse.
            # Accept with probability based on temperature?
            # Since we are minimizing max, neutral moves (same max) might help escape local minima.
            if rng.random() < 0.1 * temp:
                # Accept neutral
                update_diff_counts(new_row, 1)
                # current_max stays same
            else:
                # Reject, restore
                rows[i] = old_row
                update_diff_counts(old_row, 1)
                
        temp *= cooling

    return best_rows

# EVOLVE-BLOCK-END
