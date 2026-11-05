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
    
    Uses Simulated Annealing on row replacements to escape local minima.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def time_up():
        return time.time() > deadline

    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2

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
                # If max_limit is set, constrain the search
                # We need to leave room for remaining k-pos elements
                remaining = k - pos
                if max_limit is not None:
                    # The next element must be at most max_limit - (remaining - 1)
                    # Actually, if we pick x, the next must be > x, etc.
                    # Max possible value for current step given max_limit:
                    # x + 1 + 2 + ... + (remaining-1) <= max_limit - x? No.
                    # Simple bound: x <= max_limit - (remaining - 1)
                    max_x = max_limit - (remaining - 1)
                else:
                    max_x = 100000
                    
                # Search for valid x
                # To speed up, we can jump or just iterate.
                # Iterating from last+1 is usually fast for small k.
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
    for seed in range(50):
        if time_up(): break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_used = used

    if best_rows is None:
        # Fallback
        res = greedy_construct(0)
        if res:
            best_rows, current_used = res
            best_max = get_max(best_rows)
        else:
            return []

    # 2. Simulated Annealing
    # State: current_rows, current_used
    # Energy: max_entry + conflict_penalty (we enforce no conflicts in moves, so penalty is 0 if valid)
    # Actually, we will generate valid rows only. So energy is just max_entry.
    # To escape local minima, we allow "bad" moves (higher max) with probability exp(-delta/T).
    
    current_rows = [r[:] for r in best_rows]
    current_used = set()
    for r in current_rows:
        current_used.update(get_diffs(r))
    current_max = best_max
    
    rng = random.Random(42)
    
    # Temperature schedule
    T_start = current_max * 0.5
    T_end = 1e-6
    # We will run for a fixed number of steps or until time runs out
    # Let's estimate steps: 115s / ~1ms per step = 100k steps.
    # Greedy row generation is O(k^2 * search_width). Search width is small.
    # Let's just loop until time_up.
    
    T = T_start
    cooling_rate = 0.9999 # Adjust based on expected steps
    
    # Helper to generate a random valid row given forbidden diffs
    def generate_random_row(forbidden_diffs, rng, max_val_limit):
        row = [0]
        local_used = set()
        
        for pos in range(k):
            last = row[-1]
            remaining = k - pos
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            # Collect a few candidates to choose from, or just pick the first?
            # Picking the first is greedy. Picking random from a window is better for SA.
            # Let's collect up to 20 valid candidates in the range [last+1, last+20] or up to max_x_allowed.
            # To ensure we can find a row, we might need to search wider if blocked.
            
            valid_candidates = []
            # Search range: from last+1 to max_x_allowed
            # To keep it fast, we don't scan the whole range if we find enough.
            # But if we scan too little, we might fail.
            # Compromise: Scan up to 50 candidates or until we hit max_x_allowed.
            
            limit = min(max_x_allowed, last + 100)
            for x in range(last + 1, limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 20:
                        break
            
            if not valid_candidates:
                # If we didn't find any in the local window, try a wider scan?
                # For now, return None to indicate failure.
                return None
            
            # Pick randomly from candidates
            x, new_diffs = rng.choice(valid_candidates)
            row.append(x)
            local_used.update(new_diffs)
            
        return row

    # SA Loop
    no_improve_steps = 0
    max_no_improve = 10000
    
    while not time_up():
        # Pick a random row to replace
        i = rng.randint(0, n_rows - 1)
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove old diffs
        current_used.difference_update(old_diffs)
        
        # Determine target max for new row
        # We want to reduce the global max.
        # If this row holds the global max, try to make it smaller.
        # Otherwise, try to make it random (to reshuffle) or slightly smaller.
        
        if max(old_row) == current_max:
            # Try to improve
            target_max = current_max - 1
            # Sometimes try current_max to reshuffle
            if rng.random() < 0.1:
                target_max = current_max
        else:
            # Row is not the bottleneck.
            # Reshuffle to help others?
            # Or try to reduce it further?
            # Let's try to generate a row with max <= current_max.
            # To encourage improvement, we can bias towards smaller max.
            # But we must ensure validity.
            target_max = current_max
            # Occasionally, try to reduce this row's max below current_max?
            # It doesn't help global max directly, but might free up diffs.
            if rng.random() < 0.05:
                 target_max = max(k, max(old_row) - 1)

        new_row = generate_random_row(current_used, rng, target_max)
        
        if new_row is None:
            # Failed to generate a valid row with this limit.
            # Retry with higher limit? Or just restore.
            # To avoid getting stuck, if we failed, we can try with a higher limit (current_max)
            if target_max < current_max:
                new_row = generate_random_row(current_used, rng, current_max)
            
            if new_row is None:
                # Still failed. Restore old row.
                current_used.update(old_diffs)
                no_improve_steps += 1
                continue
        
        new_diffs = get_diffs(new_row)
        
        # Check validity (should be guaranteed by generate_random_row, but double check)
        if new_diffs & current_used:
            current_used.update(old_diffs)
            no_improve_steps += 1
            continue
            
        # Calculate new global max
        new_row_max = max(new_row)
        # The global max is max(max of other rows, new_row_max)
        # We don't store max of others explicitly, so we can compute or estimate.
        # Efficient way: if old_row was the unique max, new_max = max(new_row_max, second_max).
        # For simplicity, compute get_max(current_rows) if we replace?
        # That is O(n_rows * k). n_rows is small (4). k is small (8).
        # So O(32) operations. Very fast.
        
        # Temporarily swap
        current_rows[i] = new_row
        new_global_max = get_max(current_rows)
        
        # Acceptance Criterion
        delta = new_global_max - current_max
        
        if delta <= 0:
            # Improvement or equal
            current_used.update(new_diffs)
            current_max = new_global_max
            no_improve_steps = 0
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        else:
            # Worse
            # Accept with probability exp(-delta / T)
            if T > 1e-9:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    # Accept
                    current_used.update(new_diffs)
                    current_max = new_global_max
                    no_improve_steps += 1
                else:
                    # Reject
                    current_rows[i] = old_row
                    current_used.update(old_diffs)
                    no_improve_steps += 1
            else:
                # Reject
                current_rows[i] = old_row
                current_used.update(old_diffs)
                no_improve_steps += 1
        
        # Cooling
        T *= cooling_rate
        
        # Optional: Restart if stuck for too long?
        # The SA should handle it.

    # Final validation
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
