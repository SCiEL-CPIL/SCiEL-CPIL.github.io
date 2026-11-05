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

    def rebuild_row_lookahead(k, forbidden_diffs, rng, max_val_limit):
        """
        Build a row using greedy-with-lookahead: at each step, evaluate candidate
        gap sizes by estimating how many future difference slots they would block.
        Prefer candidates with minimal collision probability with occupied slots.
        """
        row = [0]
        local_used = set()
        
        for step in range(k):
            last = row[-1]
            remaining = k - step  # number of more elements to place (including this one)
            min_remaining_space = remaining * (remaining + 1) // 2  # min total gap needed
            
            valid_candidates = []
            search_limit = min(max_val_limit, last + max_val_limit)
            
            # Collect candidates with lookahead score
            for x in range(last + 1, search_limit + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & local_used or new_diffs & forbidden_diffs:
                    continue
                
                # Lookahead: estimate how many of these new diffs are "crowded"
                # i.e., have neighbors already used or likely to be needed
                crowd_score = 0
                for d in new_diffs:
                    # Check if nearby differences are already used
                    for delta in (-2, -1, 1, 2):
                        if (d + delta) in forbidden_diffs or (d + delta) in local_used:
                            crowd_score += 1
                
                # Also penalize large gaps that waste space
                gap_penalty = (x - last) * 0.1
                
                score = crowd_score + gap_penalty
                valid_candidates.append((x, new_diffs, score))
                
                # Early termination: if we have a good candidate and enough options
                if len(valid_candidates) >= 8:
                    break
            
            if not valid_candidates:
                return None
            
            # Sort by score (lower is better), then by x (smaller preferred)
            valid_candidates.sort(key=lambda c: (c[2], c[0]))
            
            # Pick from top candidates with some randomness
            top_n = min(3, len(valid_candidates))
            idx = rng.randint(0, top_n - 1)
            x, new_diffs, _ = valid_candidates[idx]
            
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
            
            # Check if we have enough room for remaining elements
            space_left = max_val_limit - x
            min_needed = (k - step - 1) * (k - step) // 2
            if space_left < min_needed:
                return None
        
        return row

    def rebuild_row_random(k, forbidden_diffs, rng, max_val_limit):
        """Fallback: randomized greedy without lookahead."""
        row = [0]
        local_used = set()
        
        for _ in range(k):
            last = row[-1]
            valid_candidates = []
            
            for x in range(last + 1, max_val_limit + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 5:
                        break
            
            if not valid_candidates:
                return None
            
            x, new_diffs = valid_candidates[rng.randint(0, len(valid_candidates) - 1)]
            row.append(x)
            local_used.update(new_diffs)
            
            if x > max_val_limit:
                return None
        
        return row

    rng = random.Random(12345)
    no_improve_count = 0

    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        
        old_row = rows[i]
        old_diffs = get_diffs(old_row)
        
        used_diffs.difference_update(old_diffs)
        
        new_row = None
        target_max = current_max - 1
        
        if target_max > k:
            # Try lookahead first, then random fallback
            new_row = rebuild_row_lookahead(k, used_diffs, rng, target_max)
            if new_row is None:
                new_row = rebuild_row_random(k, used_diffs, rng, target_max)
        
        if new_row is None:
            new_row = rebuild_row_lookahead(k, used_diffs, rng, current_max)
            if new_row is None:
                new_row = rebuild_row_random(k, used_diffs, rng, current_max)
        
        if new_row is None:
            used_diffs.update(old_diffs)
            no_improve_count += 1
            continue
        
        new_diffs = get_diffs(new_row)
        
        if new_diffs & used_diffs:
            used_diffs.update(old_diffs)
            no_improve_count += 1
            continue
        
        rows[i] = new_row
        new_global_max = get_max(rows)
        
        if new_global_max < current_max:
            used_diffs.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in rows]
            no_improve_count = 0
        else:
            if rng.random() < 0.15:
                used_diffs.update(new_diffs)
                no_improve_count += 1
            else:
                rows[i] = old_row
                used_diffs.update(old_diffs)
                no_improve_count += 1
        
        lb = n_rows * k * (k + 1) // 2
        if current_max == lb:
            break

    return best_rows

# EVOLVE-BLOCK-END
