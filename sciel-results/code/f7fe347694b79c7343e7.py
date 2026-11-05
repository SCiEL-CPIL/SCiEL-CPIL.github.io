import random
import time
from collections import Counter

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows, used

    rows, used_diffs = greedy_initial()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    def get_max(rows):
        return max(max(r) for r in rows)

    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def rebuild_row_fast(k, forbidden_diffs, rng, max_val_limit):
        """
        Fast row rebuild: pick smallest valid x at each step with light lookahead.
        """
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            found = False
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & local_used or new_diffs & forbidden_diffs:
                    continue
                
                # Check remaining space feasibility
                space_left = max_val_limit - x
                min_needed = (k - step - 1) * (k - step) // 2
                if space_left < min_needed:
                    break
                
                # Light lookahead: prefer x that doesn't crowd nearby diffs
                crowd = 0
                for d in new_diffs:
                    if (d - 1) in forbidden_diffs or (d + 1) in forbidden_diffs:
                        crowd += 1
                    if (d - 1) in local_used or (d + 1) in local_used:
                        crowd += 1
                
                # Accept with probability based on crowd (annealing-like)
                if crowd == 0 or rng.random() < 0.7:
                    row.append(x)
                    local_used.update(new_diffs)
                    found = True
                    break
            
            if not found:
                return None
        
        return row

    def rebuild_row_greedy_min(k, forbidden_diffs, max_val_limit):
        """Deterministic: always pick the smallest valid x."""
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            found = False
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & local_used or new_diffs & forbidden_diffs:
                    continue
                
                space_left = max_val_limit - x
                min_needed = (k - step - 1) * (k - step) // 2
                if space_left < min_needed:
                    break
                
                row.append(x)
                local_used.update(new_diffs)
                found = True
                break
            
            if not found:
                return None
        
        return row

    rng = random.Random(12345)

    # Phase 1: Main local search
    while time.time() < deadline - 15:
        i = rng.randint(0, n_rows - 1)
        
        old_row = rows[i]
        old_diffs = get_diffs(old_row)
        
        used_diffs.difference_update(old_diffs)
        
        new_row = None
        target_max = current_max - 1
        
        if target_max > k:
            # Try fast rebuild at reduced max
            new_row = rebuild_row_fast(k, used_diffs, rng, target_max)
            if new_row is None:
                new_row = rebuild_row_greedy_min(k, used_diffs, target_max)
        
        if new_row is None:
            # Try at current max
            new_row = rebuild_row_fast(k, used_diffs, rng, current_max)
            if new_row is None:
                new_row = rebuild_row_greedy_min(k, used_diffs, current_max)
        
        if new_row is None:
            used_diffs.update(old_diffs)
            continue
        
        new_diffs = get_diffs(new_row)
        
        if new_diffs & used_diffs:
            used_diffs.update(old_diffs)
            continue
        
        rows[i] = new_row
        new_global_max = get_max(rows)
        
        if new_global_max < current_max:
            used_diffs.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
        else:
            # Accept non-improving moves with small probability for exploration
            if rng.random() < 0.1:
                used_diffs.update(new_diffs)
            else:
                rows[i] = old_row
                used_diffs.update(old_diffs)
        
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    # Phase 2: Final polish - try to reduce max by 1 more
    polish_deadline = start_time + 118.0
    while time.time() < polish_deadline:
        # Find which row(s) have the max entry
        max_val = current_max
        rows_with_max = [i for i, r in enumerate(rows) if max(r) == max_val]
        
        if not rows_with_max:
            break
        
        # Try to rebuild each row with max to fit in max_val - 1
        improved = False
        for i in rows_with_max:
            if time.time() >= polish_deadline:
                break
            
            old_row = rows[i]
            old_diffs = get_diffs(old_row)
            
            used_diffs.difference_update(old_diffs)
            
            target = max_val - 1
            new_row = rebuild_row_fast(k, used_diffs, rng, target)
            if new_row is None:
                new_row = rebuild_row_greedy_min(k, used_diffs, target)
            
            if new_row is not None:
                new_diffs = get_diffs(new_row)
                if not (new_diffs & used_diffs):
                    rows[i] = new_row
                    used_diffs.update(new_diffs)
                    current_max = max(max(r) for r in rows)
                    best_rows = [r[:] for r in rows]
                    improved = True
                    break
                else:
                    used_diffs.update(old_diffs)
            else:
                used_diffs.update(old_diffs)
        
        if not improved:
            break

    return best_rows

# EVOLVE-BLOCK-END
