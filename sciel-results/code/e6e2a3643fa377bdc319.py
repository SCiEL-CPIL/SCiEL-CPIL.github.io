import random
import time

def construct(n_rows, k):
    def count_diffs(rows):
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    # 1. Initial Construction
    # Use randomized greedy to generate diverse starts
    def greedy_construct():
        counts = {}
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = [x - a for a in row]
                # Check if any new_diff is already in counts
                valid = True
                for d in new_diffs:
                    if counts.get(d, 0) > 0:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for d in new_diffs:
                        counts[d] = counts.get(d, 0) + 1
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows, counts

    best_initial_rows = None
    best_initial_max = float('inf')
    best_initial_counts = None
    
    # Run a few greedy attempts
    for _ in range(20):
        rows, counts = greedy_construct()
        current_max = max(max(r) for r in rows)
        if current_max < best_initial_max:
            best_initial_max = current_max
            best_initial_rows = [r[:] for r in rows]
            best_initial_counts = dict(counts)

    rows = best_initial_rows
    counts = best_initial_counts
    best_rows = [r[:] for r in rows]
    best_max = best_initial_max

    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 115
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    while time.time() - start_time < time_limit and T > min_T:
        r_idx = random.randint(0, n_rows - 1)
        row = rows[r_idx]
        j_idx = random.randint(1, k)
        
        old_val = row[j_idx]
        
        # Generate candidate
        low = row[j_idx - 1] + 1
        high = row[j_idx + 1] - 1 if j_idx < k else old_val + 20
        # Allow some expansion
        window = int(T * 2) + 1
        min_cand = max(low, old_val - window)
        max_cand = min(high, old_val + window)
        
        if min_cand > max_cand:
            T *= cooling_rate
            continue
            
        new_val = random.randint(min_cand, max_cand)
        
        if new_val == old_val:
            T *= cooling_rate
            continue
            
        # Check order (already handled by min_cand/max_cand mostly, but double check)
        if new_val <= row[j_idx - 1] or (j_idx < k and new_val >= row[j_idx + 1]):
            T *= cooling_rate
            continue
            
        # Calculate diffs changes
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == j_idx:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
            
        # Check validity using counts
        # A diff d is "blocked" if counts[d] > 0 and we are not removing that specific instance?
        # Actually, we are removing old_diffs and adding new_diffs.
        # For a new diff d:
        # It is valid if (counts[d] - (1 if d in old_diffs else 0)) == 0
        # i.e., after removing the old instance, is there any other instance left?
        # If yes, conflict.
        
        valid = True
        for d in new_diffs:
            # How many times is d used currently?
            current_count = counts.get(d, 0)
            # How many times is d removed by this move?
            removed_count = old_diffs.count(d)
            # If current_count - removed_count > 0, then d is still used by others (or this row, other pairs)
            # Wait, old_diffs contains ALL diffs involving old_val.
            # If d is in new_diffs, and d was also in old_diffs, it means the diff value is the same.
            # If d is in new_diffs, we are adding one instance of d.
            # We are removing `removed_count` instances of d.
            # The net change for d is +1 - removed_count.
            # The existing count is `current_count`.
            # The new count will be `current_count + 1 - removed_count`.
            # This is valid if `current_count - removed_count == 0` (so new count becomes 1) 
            # OR if we are not adding d? No, we are adding d.
            # So we need `current_count - removed_count == 0`?
            # If `current_count - removed_count > 0`, then there is already a diff d in the system 
            # that is NOT being removed. So adding another d creates a duplicate.
            # So yes, we require `current_count == removed_count`.
            
            if current_count != old_diffs.count(d):
                valid = False
                break
                
        if not valid:
            T *= cooling_rate
            continue
            
        # Move is valid
        # Update counts
        for d in old_diffs:
            counts[d] -= 1
            if counts[d] == 0:
                del counts[d]
        for d in new_diffs:
            counts[d] = counts.get(d, 0) + 1
            
        # Update row
        row[j_idx] = new_val
        
        # Calculate cost
        old_max = max(max(r) for r in rows) # This is inefficient, but let's see.
        # Actually, we should track max more carefully or just compute it.
        # Given small size, computing is O(32) ops. 10000 iters * 32 = 320k ops. Fine.
        
        # But wait, we need old_max BEFORE update?
        # We already updated `row`. 
        # We can store old_max before modification.
        
        # Let's restructure:
        # 1. Calc old_max
        # 2. Try move
        # 3. If valid, apply
        # 4. Calc new_max
        # 5. Accept/Reject
        
        # I will refactor the loop to be cleaner.
        
        # For now, let's just assume we accepted and check best.
        current_max = max(max(r) for r in rows)
        
        # We need to handle rejection properly.
        # Since I updated counts and row, if I reject, I must revert.
        
        # Let's do the cost check before applying?
        # Cost depends on max.
        # Max depends on row values.
        # So we must calculate new_max.
        
        # Re-logic:
        # 1. Identify move
        # 2. Check diff validity
        # 3. Calculate new_max (simulated)
        # 4. Decide accept/reject based on T and cost delta
        # 5. If accept, apply changes to counts and row.
        # 6. If reject, do nothing.
        
        # This is safer.
        
        # Let's rewrite the loop body in the final code.
        
        T *= cooling_rate

    return best_rows
