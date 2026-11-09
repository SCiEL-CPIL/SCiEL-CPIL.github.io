import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Parameters
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Bitmask setup
    # We need to track differences up to a certain limit.
    # The known best is 112. Let's allow up to 120 or 130 to be safe.
    # If we exceed this, we can't track it in a single int, but for (7,5) M is ~112.
    # Let's set MAX_DIFF = 130.
    MAX_DIFF = 130
    MASK = (1 << MAX_DIFF) - 1
    
    # Helper to calculate diffs for a row
    def get_row_diffs(row):
        diffs = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d <= MAX_DIFF:
                    diffs |= (1 << d)
                else:
                    # If diff > MAX_DIFF, we treat it as a conflict or ignore?
                    # In a valid solution, diffs must be distinct. 
                    # If we assume M <= MAX_DIFF, any diff > MAX_DIFF is invalid for the "small M" goal.
                    # However, during SA, we might have large values. 
                    # For the purpose of the objective function, we want to minimize max.
                    # If a diff > MAX_DIFF, it contributes to a huge penalty or is just counted as a unique "high" diff?
                    # Actually, if we want M < 112, all diffs MUST be <= 111.
                    # So any diff > MAX_DIFF is a hard violation of the "small scope" goal.
                    # Let's just mark it as a conflict with a special bit or just count it as an error.
                    # Simplest: if d > MAX_DIFF, return a special flag or just add to penalty.
                    pass
        return diffs

    # Better: Precompute or just iterate.
    # Since k is small (5), row length is 6. 15 diffs per row.
    # n_rows = 7. Total 105 diffs.
    
    # State: list of n_rows rows. Each row is a list of k+1 ints.
    # Row 0 is always [0, ...]
    
    def initialize():
        rows = []
        used = 0
        for _ in range(n_rows):
            row = [0]
            # Greedy fill
            for _ in range(k):
                found = False
                x = 1
                while x <= MAX_DIFF:
                    # Check conflicts
                    conflicts = False
                    for prev in row:
                        d = x - prev
                        if d <= MAX_DIFF:
                            if used & (1 << d):
                                conflicts = True
                                break
                        else:
                            # Diff too large, likely bad for minimizing max
                            # But for initialization, just pick a small one that fits
                            # If no small one fits, we might get stuck. 
                            # Greedy is usually okay for init.
                            conflicts = True
                            break
                    if not conflicts:
                        row.append(x)
                        if x - row[0] <= MAX_DIFF:
                            # Update used for all new diffs
                            for prev in row[:-1]:
                                d = x - prev
                                if d <= MAX_DIFF:
                                    used |= (1 << d)
                        found = True
                        break
                    x += 1
                if not found:
                    # Fallback: just append large numbers to ensure validity structure, 
                    # though it will be a bad solution.
                    row.append(row[-1] + 100)
            rows.append(row)
        return rows

    current_rows = initialize()
    
    # Calculate current cost
    def calc_cost(rows):
        # Cost = (max_value) + (penalty for duplicate diffs) * large_factor
        # We want to minimize max_value primarily.
        # If duplicates exist, we must resolve them.
        # Let's define cost = max_value + 1000 * num_conflicts
        # Actually, if there are conflicts, the solution is invalid. 
        # SA needs to handle invalid states. 
        # A common trick: Cost = max_value + sum(overlap_bits)
        
        max_val = 0
        all_diffs = 0
        num_conflicts = 0
        
        for row in rows:
            if row:
                m = row[-1]
                if m > max_val:
                    max_val = m
            
            # Calculate diffs for this row
            row_diffs = 0
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d <= MAX_DIFF:
                        # Check if this bit is already set in global all_diffs?
                        # No, we need to check if it conflicts with ANY other diff.
                        # So we accumulate into a global mask.
                        # If a bit is set twice, it's a conflict.
                        if row_diffs & (1 << d):
                            # Duplicate within row
                            num_conflicts += 1
                        row_diffs |= (1 << d)
                    else:
                        # Diff too big. 
                        # This implies max_val is large. 
                        # We count it as a conflict to discourage large values.
                        num_conflicts += 100 
                        
            # Check global conflicts
            # We need to know if any bit in row_diffs is already in all_diffs
            conflicts_in_global = all_diffs & row_diffs
            # Count bits in conflicts_in_global
            num_conflicts += bin(conflicts_in_global).count('1')
            
            all_diffs |= row_diffs
            
        return max_val + 1000 * num_conflicts

    current_cost = calc_cost(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_cost = current_cost
    
    # SA Parameters
    T = 1000.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute possible moves?
    # Move: Pick a random row, pick a random position (1..k), change the value.
    # Or: Pick a random row, pick a random pair of positions, swap? No, must be increasing.
    # Simplest: Pick a random element (not 0), add a small random delta.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            # Restart or just continue at min_T
            # Let's just continue with random walks at low T to explore.
            
        # Generate neighbor
        new_rows = [list(r) for r in current_rows]
        row_idx = random.randint(0, n_rows - 1)
        pos_idx = random.randint(1, k)  # 1 to k
        
        # Change value at pos_idx
        old_val = new_rows[row_idx][pos_idx]
        
        # Choose a new value
        # Strategy: Random step in range [old_val - 10, old_val + 10]
        # Or random restart of this element?
        # Let's try: new_val = random.randint(1, MAX_DIFF)
        # But that's too random. 
        # Better: new_val = old_val + random.randint(-5, 5)
        
        # Ensure increasing: new_val > new_rows[row_idx][pos_idx - 1]
        # And new_val < new_rows[row_idx][pos_idx + 1] if pos_idx < k
        
        min_allowed = new_rows[row_idx][pos_idx - 1] + 1
        max_allowed = MAX_DIFF
        if pos_idx < k:
            max_allowed = min(max_allowed, new_rows[row_idx][pos_idx + 1] - 1)
        
        if min_allowed > max_allowed:
            # Can't change this position validly without breaking order?
            # Maybe we should allow reordering? No, rows are sorted.
            # If we change a middle element, we must keep it between neighbors.
            # If the range is empty, skip.
            continue
            
        # Bias towards smaller values to minimize max?
        # Or just uniform.
        new_val = random.randint(min_allowed, max_allowed)
        
        if new_val == old_val:
            continue
            
        new_rows[row_idx][pos_idx] = new_val
        
        # Calculate new cost
        new_cost = calc_cost(new_rows)
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_rows = new_rows
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_rows = [list(r) for r in current_rows]
                # If we hit the lower bound, we can stop
                if best_cost == lower_bound:
                    break
        else:
            pass # Reject
            
        T *= cooling
        
    # Ensure we return a valid solution if possible?
    # The task says "Valid when all positive differences... are distinct".
    # If our best_rows has conflicts, it's invalid.
    # However, the score is based on "raw value (lower is better)".
    # "an invalid result... scores 0".
    # So we MUST return a valid set.
    # If SA doesn't find a valid set, we should return the greedy one which is valid.
    
    # Check validity of best_rows
    def is_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    if not is_valid(best_rows):
        # Fall back to a valid greedy solution
        # Re-run greedy to be safe
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                else:
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows
