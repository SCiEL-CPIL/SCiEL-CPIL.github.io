import time
import random
import copy
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 140.0
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initial valid solution
    best_rows = greedy_construct()
    best_scope = max(max(row) for row in best_rows)
    
    current_rows = copy.deepcopy(best_rows)
    
    def calculate_initial_state(rows):
        diff_counter = Counter()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff_counter[row[j] - row[i]] += 1
        conflicts = sum(c - 1 for c in diff_counter.values() if c > 1)
        scope = max(max(row) for row in rows)
        return diff_counter, conflicts, scope

    diff_counter, curr_conflicts, curr_scope = calculate_initial_state(current_rows)
    
    def get_scope(rows):
        return max(max(row) for row in rows)

    # Function to try to shrink a specific element to the smallest valid value
    # Returns True if a change was made
    def try_shrink_element(r, j):
        nonlocal curr_scope, curr_conflicts, best_scope, best_rows
        row = current_rows[r]
        old_val = row[j]
        
        # The new value must be > row[j-1]
        min_possible = row[j-1] + 1
        
        # If old_val is already at min, we can't shrink
        if old_val == min_possible:
            return False
            
        # We want to find the smallest v in [min_possible, old_val - 1] that is valid.
        # We check candidates from low to high.
        # However, checking every single integer might be slow if the gap is large.
        # But typically, valid values are sparse.
        # Let's try a few candidates. If we find one, we take it.
        # To be efficient, we can just try to decrement by 1 first? 
        # No, the directive says "replace with the smallest integer that restores property".
        # So we should search for the minimum valid v.
        
        # To avoid O(N^2) checks for each v (where N is row size), we precompute
        # the differences that would be created by v.
        
        # Optimization: 
        # The differences involving index j are:
        # v - row[i] for i < j
        # row[i] - v for i > j
        
        # We need to ensure none of these new differences conflict with existing ones
        # (after removing the old ones).
        
        # Let's remove old diffs from counter
        old_diffs = []
        for i in range(j):
            d = old_val - row[i]
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
                
        for i in range(j+1, len(row)):
            d = row[i] - old_val
            old_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        # Now search for smallest valid v
        found = False
        new_val = -1
        
        # We iterate from min_possible upwards.
        # If we find a valid one, we use it.
        # If the range is huge, this is slow. But we only do this if we can shrink.
        # If old_val is large, the range is large.
        # However, we only call this if we are in a "shrink" phase.
        # Let's limit the search depth if it gets too expensive? 
        # Or just try a few steps down from old_val?
        # The directive suggests "smallest integer". Let's stick to that but be careful with time.
        
        # Heuristic: Check candidates from min_possible up to old_val-1.
        # If we find a valid one, we stop.
        # If we don't find one quickly, we might revert.
        
        # To speed up, we can check if v is valid by seeing if any of its diffs are in diff_counter.
        
        # We'll try to find the first valid v.
        # If the gap is > 1000, maybe just try old_val - 1, old_val - 2...?
        # No, "smallest" is key to minimizing scope.
        
        # Let's try a hybrid: 
        # 1. Try v = min_possible.
        # 2. If invalid, try v = min_possible + 1, etc.
        # But if min_possible is 10 and old_val is 100, and 10..50 are all invalid, 
        # and 51 is valid, we should take 51.
        
        # Given the small k (8), the number of diffs per element is small (<= 15).
        # Checking validity is O(k).
        # So searching 100 candidates is O(100*k) = O(1500) ops. Fast.
        
        # We'll search up to a certain limit or until found.
        # If we find a valid v < old_val, we apply it.
        
        # To avoid getting stuck in a loop where we can't find a valid v immediately,
        # we'll just scan from min_possible.
        
        # However, scanning from 1 to 100 every time might be wasteful if 1 is always invalid.
        # But correctness first.
        
        # Let's add a timeout check inside the loop if the range is large.
        limit = old_val - 1
        # If range is very large, we might just check a few values near old_val?
        # No, that defeats "smallest".
        # Let's just scan. It's bounded by old_val.
        
        for v in range(min_possible, old_val):
            # Check validity of v
            valid = True
            new_diffs = []
            for i in range(j):
                d = v - row[i]
                new_diffs.append(d)
                if d in diff_counter:
                    valid = False
                    break
            
            if not valid:
                continue
                
            for i in range(j+1, len(row)):
                d = row[i] - v
                new_diffs.append(d)
                if d in diff_counter:
                    valid = False
                    break
                    
            if valid:
                # Apply v
                row[j] = v
                new_val = v
                for d in new_diffs:
                    diff_counter[d] += 1
                found = True
                break
            # else: v is invalid, keep searching
            
        if not found:
            # Revert: re-add old diffs
            for d in old_diffs:
                diff_counter[d] += 1
            # row[j] is still old_val
            return False
        else:
            # Update scope and conflicts
            # Conflicts should be 0 if we started with 0 and only made valid moves.
            # But let's update curr_scope.
            new_scope = get_scope(current_rows)
            if new_scope < curr_scope:
                curr_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
            # curr_conflicts remains 0 (assuming we started valid and moved validly)
            return True

    # Main Loop
    # We alternate between:
    # 1. Systematic Shrink: Go through all elements (r, j) and try to shrink to min valid.
    # 2. Perturbation: If no shrink found in a full pass, do a random kick to escape local min.
    
    # Order of (r, j) to prioritize scope reduction:
    # Sort by potential impact. Last elements of rows are most important for scope.
    # Let's just iterate r in 0..n_rows-1, j in k..1.
    
    no_improve_count = 0
    max_no_improve = 100 # After this many full passes without improvement, kick harder
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Systematic Shrink
        for r in range(n_rows):
            # Check time
            if time.time() > deadline:
                break
            # Iterate j from k down to 1
            # We might want to prioritize j=k (the max element) first
            for j in range(k, 0, -1):
                if try_shrink_element(r, j):
                    improved = True
                    # If we shrank, we might want to re-check this row or continue?
                    # Continuing is fine, as shrinking one element might open up space for others
                    # or close it. But since we are doing a full pass, it will be checked again next pass.
                    # However, shrinking element j might make element j+1 shrinkable if it was blocked?
                    # No, j+1 is larger. Shrinking j (smaller index) doesn't directly unblock j+1's lower bound 
                    # (which is row[j]+1). If row[j] decreases, row[j]+1 decreases, so j+1 has MORE room to shrink.
                    # So it's good to continue.
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If we are stuck, perform a perturbation
        if no_improve_count > 0:
            # Random Kick
            # Pick a random element and move it to a random valid position (possibly larger)
            # to escape local minimum.
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            
            old_val = current_rows[r][j]
            row = current_rows[r]
            
            min_val = row[j-1] + 1
            # Allow expansion to escape
            max_val = old_val + 50 
            if j < k:
                max_val = min(max_val, row[j+1] - 1)
            
            if min_val > max_val:
                continue
                
            # Pick a random new value different from old
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                continue
                
            # Check validity of new_val
            # Remove old diffs
            old_diffs = []
            for i in range(j):
                d = old_val - row[i]
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for i in range(j+1, len(row)):
                d = row[i] - old_val
                old_diffs.append(d)
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            # Check new diffs
            valid = True
            new_diffs = []
            for i in range(j):
                d = new_val - row[i]
                new_diffs.append(d)
                if d in diff_counter:
                    valid = False
                    break
            if valid:
                for i in range(j+1, len(row)):
                    d = row[i] - new_val
                    new_diffs.append(d)
                    if d in diff_counter:
                        valid = False
                        break
            
            if valid:
                # Apply
                row[j] = new_val
                for d in new_diffs:
                    diff_counter[d] += 1
                curr_scope = get_scope(current_rows)
                # curr_conflicts is still 0
                # Reset no_improve to give it a chance
                no_improve_count = 0 
            else:
                # Revert
                for d in old_diffs:
                    diff_counter[d] += 1
                # row[j] remains old_val

    return best_rows

# Quick sanity check / self-test if run directly
if __name__ == "__main__":
    rows = construct(4, 7)
    scope = max(max(r) for r in rows)
    print(f"Scope: {scope}")
    # Verify validity
    diffs = set()
    valid = True
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print(f"Valid: {valid}")
    print(rows)
