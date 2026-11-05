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
    
    Strategy:
    1. Fast randomized greedy to get a valid baseline.
    2. Local search with row-rebuild to tighten the maximum entry.
    3. Final polish pass: targeted rebuilds of rows holding the current max,
       with tighter candidate search and multiple restarts.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit

    def time_up():
        return time.time() > deadline

    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    # Helper to compute all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Multi-start Greedy
    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for _ in range(k):
                last = row[-1]
                valid_candidates = []
                
                width = 5
                
                for x in range(last + 1, 100000):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None
                
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
                
                row.append(x)
                local_used.update(new_diffs)
                
            rows.append(row)
            used.update(local_used)
            
        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None
    
    for seed in range(20):
        if time_up(): break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used = used
    
    if best_rows is None:
        res = greedy_construct(0)
        if res:
            best_rows, current_used = res
            current_rows = best_rows
            best_max = get_max(best_rows)
        else:
            return []

    if best_max == lb:
        return best_rows

    # 2. Local Search with Rebuilds
    rng = random.Random(42)
    current_max = best_max
    
    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.0, width=10):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        bias: probability to pick a random valid candidate from the first `width`, else smallest.
        """
        row = [0]
        local_used = set()
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            remaining = k - len(row)
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            for x in range(last + 1, max_x_allowed + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= width:
                        break
            
            if not valid_candidates:
                return None
            
            if bias > 0 and rng.random() < bias:
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
            else:
                x, new_diffs = valid_candidates[0]
            
            row.append(x)
            local_used.update(new_diffs)
            
        return row

    def rebuild_row_tight(k, forbidden_diffs, rng, max_val_limit, width=5):
        """
        Tighter rebuild: smaller candidate window, more deterministic.
        Used in the final polish pass.
        """
        row = [0]
        local_used = set()
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            remaining = k - len(row)
            max_x_allowed = max_val_limit - (remaining - 1)
            
            if last + 1 > max_x_allowed:
                return None
            
            for x in range(last + 1, max_x_allowed + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= width:
                        break
            
            if not valid_candidates:
                return None
            
            # Deterministic: always pick smallest
            x, new_diffs = valid_candidates[0]
            
            row.append(x)
            local_used.update(new_diffs)
            
        return row

    no_improve_count = 0
    max_no_improve = 100
    
    while not time_up():
        i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        current_used.difference_update(old_diffs)
        
        new_row = None
        
        if max(old_row) == current_max:
            target_max = current_max - 1
            if target_max > k:
                new_row = rebuild_row(k, current_used, rng, target_max, bias=0.5, width=10)
        
        if new_row is None:
            new_row = rebuild_row(k, current_used, rng, current_max, bias=0.8, width=10)
        
        if new_row is None:
            current_used.update(old_diffs)
            no_improve_count += 1
        else:
            current_rows[i] = new_row
            new_diffs = get_diffs(new_row)
            current_used.update(new_diffs)
            
            new_global_max = get_max(current_rows)
            if new_global_max < current_max:
                current_max = new_global_max
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            no_improve_count = 0

    # 3. Final polish pass: targeted rebuilds of rows holding the current max
    #    with tighter tolerance and multiple restarts.
    if not time_up() and current_max > lb:
        polish_rng = random.Random(12345)
        polish_deadline = time.time() + 20.0  # Use up to 20s for polish
        
        for polish_attempt in range(30):
            if time.time() > polish_deadline or time_up():
                break
            
            # Identify rows that achieve the current max
            max_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_max]
            
            if not max_rows:
                break
            
            # Try to rebuild one of the max-holding rows with max_val_limit = current_max - 1
            i = polish_rng.choice(max_rows)
            old_row = current_rows[i]
            old_diffs = get_diffs(old_row)
            
            current_used.difference_update(old_diffs)
            
            target_max = current_max - 1
            if target_max <= k:
                current_used.update(old_diffs)
                break
            
            # Try multiple rebuilds with different seeds
            new_row = None
            for attempt in range(5):
                if time.time() > polish_deadline:
                    break
                # Alternate between tight and slightly randomized
                if attempt % 2 == 0:
                    new_row = rebuild_row_tight(k, current_used, polish_rng, target_max, width=5)
                else:
                    new_row = rebuild_row(k, current_used, polish_rng, target_max, bias=0.3, width=7)
                
                if new_row is not None:
                    break
            
            if new_row is not None:
                current_rows[i] = new_row
                new_diffs = get_diffs(new_row)
                current_used.update(new_diffs)
                
                new_global_max = get_max(current_rows)
                if new_global_max < current_max:
                    current_max = new_global_max
                else:
                    # Roll back if no improvement
                    current_rows[i] = old_row
                    current_used.difference_update(new_diffs)
                    current_used.update(old_diffs)
            else:
                current_used.update(old_diffs)
        
        # After polish, do a few more general local search steps if time permits
        if not time_up() and current_max > lb:
            remaining_time = deadline - time.time()
            if remaining_time > 5.0:
                # Quick final local search round
                for _ in range(50):
                    if time.time() > deadline - 2.0:
                        break
                    
                    i = rng.randint(0, n_rows - 1)
                    old_row = current_rows[i]
                    old_diffs = get_diffs(old_row)
                    
                    current_used.difference_update(old_diffs)
                    
                    new_row = None
                    if max(old_row) == current_max:
                        target_max = current_max - 1
                        if target_max > k:
                            new_row = rebuild_row_tight(k, current_used, rng, target_max, width=5)
                    
                    if new_row is None:
                        new_row = rebuild_row(k, current_used, rng, current_max, bias=0.6, width=8)
                    
                    if new_row is None:
                        current_used.update(old_diffs)
                    else:
                        current_rows[i] = new_row
                        new_diffs = get_diffs(new_row)
                        current_used.update(new_diffs)
                        
                        new_global_max = get_max(current_rows)
                        if new_global_max < current_max:
                            current_max = new_global_max

    # Validate
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
    
    if not validate(current_rows, n_rows, k):
        return best_rows
    
    return current_rows
# EVOLVE-BLOCK-END
