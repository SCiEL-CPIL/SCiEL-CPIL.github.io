import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2

    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True

    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_solution(seed, target_max=None):
        random.seed(seed)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                x = current_max + 1
                if target_max is not None:
                    search_limit = target_max
                else:
                    search_limit = x + 500 + row_idx * 50
                
                candidates = []
                while x <= search_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        candidates.append(x)
                        if len(candidates) > 5:
                            break
                    x += 1
                
                if not candidates:
                    if target_max is not None:
                        return None
                    else:
                        # Unbounded search
                        x = current_max + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used) and len(new_diffs) == len(row):
                                candidates.append(x)
                                break
                            x += 1
                            if time.time() - start_time > time_limit - 10:
                                return None
                
                candidates.sort()
                # Randomized choice to create diversity
                # Pick from top few
                top_cands = candidates[:min(10, len(candidates))]
                chosen = random.choice(top_cands)
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
            
            if time.time() - start_time > time_limit - 20:
                return None
        
        return rows

    def local_search_max_shrink(rows, time_budget):
        """
        Structural Change: Instead of generic Simulated Annealing, use a targeted
        local search that specifically tries to shrink the current maximum element.
        This is more efficient for minimizing the scope.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        used_diffs = set()
        row_diffs = []
        for r in rows:
            rd = compute_row_diffs(r)
            row_diffs.append(rd)
            used_diffs.update(rd)
            
        current_score = best_score
        end_time = time.time() + time_budget
        
        # We will try to shrink the global max repeatedly.
        # If we get stuck, we might need to move other elements to free up space.
        # But first, let's just try to shrink the max.
        
        no_improve = 0
        
        while time.time() < end_time:
            # Find the global max
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_val == 0:
                break
            
            row = rows[max_r]
            current_val = row[max_i]
            
            # Try to find a smaller valid value for this position
            min_val = row[max_i - 1] + 1 if max_i > 0 else 1
            max_allowed = current_val - 1
            if max_i < len(row) - 1:
                max_allowed = min(max_allowed, row[max_i + 1] - 1)
            
            if max_allowed < min_val:
                # Cannot shrink this element. 
                # Strategy: Try to shrink a *different* element to open up space?
                # Or just accept we are stuck and try a different random perturbation?
                # For now, let's try to shrink the *second* largest element?
                # Or just break and let the outer loop try a different seed?
                # Actually, if we can't shrink the max, we are likely at a local minimum for this specific structure.
                # Let's try to move a random element to a slightly different value to "shake" the structure.
                # This is a form of random restart within the local search.
                
                # Pick a random element (not the max)
                r_idx = random.randint(0, n_rows - 1)
                # Avoid index 0
                if k_plus_1 > 1:
                    j_idx = random.randint(1, k)
                else:
                    break
                
                row = rows[r_idx]
                old_val = row[j_idx]
                
                # Try to move it
                low_bound = row[j_idx - 1] + 1
                if j_idx < k:
                    high_bound = row[j_idx + 1] - 1
                else:
                    high_bound = old_val + 10
                
                if low_bound > high_bound:
                    continue
                    
                # Pick a new value
                # Bias towards smaller values if possible
                v = random.randint(low_bound, high_bound)
                
                if v == old_val:
                    continue

                # Check validity
                old_diffs_to_remove = set()
                for i in range(j_idx):
                    old_diffs_to_remove.add(old_val - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    old_diffs_to_remove.add(row[i] - old_val)
                
                new_diffs_to_add = set()
                for i in range(j_idx):
                    new_diffs_to_add.add(v - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    new_diffs_to_add.add(row[i] - v)
                    
                if len(new_diffs_to_add) != k:
                    continue
                
                conflict = False
                for d in new_diffs_to_add:
                    if d in used_diffs:
                        if d not in old_diffs_to_remove:
                            conflict = True
                            break
                
                if conflict:
                    continue
                
                # Apply
                row[j_idx] = v
                used_diffs.difference_update(old_diffs_to_remove)
                used_diffs.update(new_diffs_to_add)
                row_diffs[r_idx] = new_diffs_to_add | (row_diffs[r_idx] - old_diffs_to_remove)
                
                new_score = get_score(rows)
                
                # Always accept if it doesn't increase the score?
                # Or use a simple hill climbing: accept if better, reject if worse.
                # Since we are shaking, we might accept worse temporarily?
                # Let's just accept if it's the same or better, to avoid getting stuck.
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject, undo
                    row[j_idx] = old_val
                    used_diffs.difference_update(new_diffs_to_add)
                    used_diffs.update(old_diffs_to_remove)
                    row_diffs[r_idx] = old_diffs_to_remove | (row_diffs[r_idx] - new_diffs_to_add)
                    no_improve += 1
                    
                if no_improve > 1000:
                    # Stuck, break
                    break
                continue
            
            # If we can shrink the max, try to do so
            # Search for a valid v in [min_val, max_allowed]
            # We want the smallest v.
            # But sometimes a slightly larger v might be better for future moves?
            # Let's just try to find *any* valid v < current_val.
            # To speed up, we can sample or just iterate from min_val up.
            
            found = False
            # Limit search depth to avoid timeout on huge gaps
            # If the gap is huge, iterating is slow.
            # Let's try a few random candidates first, then linear search from bottom?
            # Actually, linear search from min_val is best for minimizing max.
            
            # Optimization: Only search if the gap is small enough.
            # If gap > 500, just pick a random candidate in the middle?
            # No, we want to shrink.
            
            # Let's iterate from min_val to max_allowed, but cap the number of steps.
            # If we don't find one in 1000 steps, we might be stuck.
            
            old_diffs_to_remove = set()
            for i in range(max_i):
                old_diffs_to_remove.add(current_val - row[i])
            for i in range(max_i + 1, k_plus_1):
                old_diffs_to_remove.add(row[i] - current_val)
                
            used_without_old = used_diffs - old_diffs_to_remove
            
            # Try candidates
            # We can't iterate too many.
            # Let's try a random sample of candidates first?
            # Or just linear search from min_val.
            
            # If the range is large, linear search is O(N).
            # Let's try to be smart: 
            # If we just want *any* reduction, we can try the midpoint?
            # But we want the *smallest* max.
            
            # Let's try linear search from min_val, but break if time runs out.
            # Also, we can skip values that are obviously bad?
            # Hard to know.
            
            # Let's just do a limited linear search.
            search_limit = min(max_allowed, min_val + 200)
            
            for v in range(min_val, search_limit + 1):
                new_diffs_to_add = set()
                valid = True
                for i in range(max_i):
                    d = v - row[i]
                    if d in used_without_old:
                        valid = False
                        break
                    new_diffs_to_add.add(d)
                if not valid:
                    continue
                for i in range(max_i + 1, k_plus_1):
                    d = row[i] - v
                    if d in used_without_old:
                        valid = False
                        break
                    new_diffs_to_add.add(d)
                
                if valid and len(new_diffs_to_add) == k:
                    # Found a valid smaller value
                    # Apply
                    row[max_i] = v
                    used_diffs.difference_update(old_diffs_to_remove)
                    used_diffs.update(new_diffs_to_add)
                    row_diffs[max_r] = new_diffs_to_add | (row_diffs[max_r] - old_diffs_to_remove)
                    
                    new_score = get_score(rows)
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    found = True
                    break
            
            if not found:
                # Couldn't shrink the max in the searched range.
                # This triggers the "shake" logic above in the next iteration
                # because max_allowed < min_val is false, but we just failed to find a v.
                # So we need to handle the "failed to shrink" case by shaking.
                # The logic above `if max_allowed < min_val` only handles the case where the range is empty.
                # If the range is not empty but no valid v is found, we are also stuck.
                # Let's add a shake here too.
                
                # Pick a random element to move
                r_idx = random.randint(0, n_rows - 1)
                if k_plus_1 > 1:
                    j_idx = random.randint(1, k)
                else:
                    break
                
                row = rows[r_idx]
                old_val = row[j_idx]
                
                low_bound = row[j_idx - 1] + 1
                if j_idx < k:
                    high_bound = row[j_idx + 1] - 1
                else:
                    high_bound = old_val + 10
                
                if low_bound > high_bound:
                    no_improve += 1
                    continue
                    
                v = random.randint(low_bound, high_bound)
                
                if v == old_val:
                    no_improve += 1
                    continue

                old_diffs_to_remove = set()
                for i in range(j_idx):
                    old_diffs_to_remove.add(old_val - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    old_diffs_to_remove.add(row[i] - old_val)
                
                new_diffs_to_add = set()
                for i in range(j_idx):
                    new_diffs_to_add.add(v - row[i])
                for i in range(j_idx + 1, k_plus_1):
                    new_diffs_to_add.add(row[i] - v)
                    
                if len(new_diffs_to_add) != k:
                    no_improve += 1
                    continue
                
                conflict = False
                for d in new_diffs_to_add:
                    if d in used_diffs:
                        if d not in old_diffs_to_remove:
                            conflict = True
                            break
                
                if conflict:
                    no_improve += 1
                    continue
                
                # Apply
                row[j_idx] = v
                used_diffs.difference_update(old_diffs_to_remove)
                used_diffs.update(new_diffs_to_add)
                row_diffs[r_idx] = new_diffs_to_add | (row_diffs[r_idx] - old_diffs_to_remove)
                
                new_score = get_score(rows)
                
                if new_score <= current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject, undo
                    row[j_idx] = old_val
                    used_diffs.difference_update(new_diffs_to_add)
                    used_diffs.update(old_diffs_to_remove)
                    row_diffs[r_idx] = old_diffs_to_remove | (row_diffs[r_idx] - new_diffs_to_add)
                    no_improve += 1
                    
                if no_improve > 1000:
                    break

        return best_rows

    # Main Logic
    best_rows = None
    best_score = float('inf')
    
    # Phase 1: Get valid solutions
    # Try a few seeds
    for seed in range(20):
        if time.time() - start_time > time_limit - 30:
            break
        rows = greedy_solution(seed)
        if rows and is_valid(rows):
            score = get_score(rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Phase 2: Local Search (Max Shrink + Shake)
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        # Run local search for a portion of remaining time
        # Split into a few runs to allow restarts?
        # The local search function has its own restart logic (shaking).
        # Let's just run it for most of the time.
        ls_time = remaining * 0.9
        final_rows = local_search_max_shrink(best_rows, ls_time)
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            best_score = final_score

    return best_rows
