import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # 1. Greedy Initial Solution
    chosen = []
    counts = [0] * (max_sum + 1)
    
    for x in range(1, N + 1):
        # Check if adding x violates B2[2]
        # Sums involving x: x + y for y in chosen, and 2x
        valid = True
        # We need to check if any sum becomes 3
        # Since we only add if valid, current max count is 2.
        # Adding x increases counts for sums x+y and 2x by 1.
        # So we fail if any of these sums currently has count 2.
        
        # Optimization: Pre-compute sums? No, just check.
        # The number of chosen elements is ~250.
        for y in chosen:
            s = x + y
            if counts[s] == 2:
                valid = False
                break
        if valid and counts[2 * x] == 2:
            valid = False
            
        if valid:
            # Add x
            chosen.append(x)
            counts[2 * x] += 1
            for y in chosen[:-1]:
                counts[x + y] += 1
                
    best_set = set(chosen)
    best_size = len(chosen)
    
    # 2. Local Search (Simulated Annealing)
    # State: set S, counts array
    # We work with a list for fast iteration and a set for membership?
    # List is faster for sum updates. Set is O(1) for membership.
    # Let's use a list `current` and a boolean array `in_set` for O(1) check.
    
    current = list(chosen)
    in_set = [False] * (N + 1)
    for x in current:
        in_set[x] = True
        
    # Re-initialize counts from current set to be safe (though greedy counts should be correct)
    counts = [0] * (max_sum + 1)
    for i in range(len(current)):
        x = current[i]
        counts[2 * x] += 1
        for j in range(i + 1, len(current)):
            y = current[j]
            counts[x + y] += 1
            
    # Calculate initial conflicts (should be 0)
    def get_conflicts(counts):
        c = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                c += counts[s] - 2
        return c

    current_conflicts = get_conflicts(counts)
    if current_conflicts > 0:
        # Fallback if greedy failed (unlikely)
        print("Greedy produced conflicts?", current_conflicts)
        
    best_conflicts = current_conflicts
    best_solution = list(current)
    
    T = 1.0
    cooling = 0.999
    
    # Precompute neighbors? No, random sampling is fine.
    # To speed up, we can precompute for each x, the list of sums it creates?
    # No, the count update is O(|S|) which is ~250-300. This is fast.
    
    while time.time() < deadline:
        # Move Generation
        # 50% Add, 50% Swap
        if random.random() < 0.8:
            # Try Add
            x = random.randint(1, N)
            if in_set[x]:
                continue
            
            # Calculate delta conflicts
            # Adding x increases counts for 2x and x+y for all y in current
            # We need to know how many of these sums are currently at 2 (becoming 3)
            # and how many are at 1 (becoming 2, ok) or 0 (becoming 1, ok)
            
            delta_conf = 0
            # Check 2x
            if counts[2 * x] == 2:
                delta_conf += 1
            # Check x+y
            for y in current:
                s = x + y
                if counts[s] == 2:
                    delta_conf += 1
            
            # Acceptance
            if delta_conf == 0:
                # Accept, update state
                in_set[x] = True
                current.append(x)
                counts[2 * x] += 1
                for y in current[:-1]:
                    counts[x + y] += 1
                current_conflicts += delta_conf
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_solution = list(current)
                        if len(current) > best_size:
                            best_size = len(current)
            else:
                # Delta > 0
                # Accept with prob exp(-delta / T)
                if random.random() < math.exp(-delta_conf / T):
                    in_set[x] = True
                    current.append(x)
                    counts[2 * x] += 1
                    for y in current[:-1]:
                        counts[x + y] += 1
                    current_conflicts += delta_conf
        
        else:
            # Try Swap
            if not current:
                continue
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            
            y = random.randint(1, N)
            if in_set[y]:
                continue
                
            # Calculate delta conflicts for removing x and adding y
            # This is slightly more complex.
            # 1. Remove x:
            #    Counts decrease for 2x and x+z for z in current (z != x)
            # 2. Add y:
            #    Counts increase for 2y and y+z for z in current (z != x, since x is removed)
            
            # We can simulate this on a temporary copy of counts?
            # No, that's O(N). We can calculate delta directly.
            
            # Let's just do it step-by-step on the actual counts, and rollback if rejected.
            # Or calculate delta first.
            
            # Delta from removing x:
            # Sums involving x: 2x, x+z (z in S, z!=x)
            # For each sum s, count decreases by 1.
            # If count was 3, it becomes 2 (conflict resolved, delta -1)
            # If count was 2, it becomes 1 (no change in conflict count, delta 0)
            # If count was 1, it becomes 0 (no change)
            
            # Delta from adding y:
            # Sums involving y: 2y, y+z (z in S \ {x})
            # For each sum s, count increases by 1.
            # If count was 2, it becomes 3 (new conflict, delta +1)
            # If count was 1, it becomes 2 (no change)
            
            # Note: The set S changes. When adding y, z iterates over S \ {x}.
            
            delta_conf = 0
            
            # 1. Effect of removing x
            # Sums: 2x
            if counts[2 * x] == 3:
                delta_conf -= 1
            # Sums: x + z for z in current, z != x
            for z in current:
                if z == x: continue
                s = x + z
                if counts[s] == 3:
                    delta_conf -= 1
            
            # 2. Effect of adding y
            # We must consider the counts *after* removing x.
            # Let c'(s) = counts(s) - (1 if s involves x else 0)
            # If c'(s) == 2, then adding y makes it 3.
            
            # Check 2y
            c_2y = counts[2 * y]
            # Does 2y involve x? i.e., 2y == 2x or 2y == x+z?
            # 2y == 2x => y == x (impossible, y not in set, x in set)
            # 2y == x + z => x = 2y - z.
            # This is getting complicated to check on the fly for each sum.
            
            # Simpler approach:
            # Temporarily update counts, check conflicts, rollback if needed.
            # But checking all conflicts is O(N).
            # We only need to check sums involving x (removed) and y (added).
            
            # Let's just do the update and track the specific sums.
            
            sums_to_check_remove = set()
            sums_to_check_add = set()
            
            # Sums removed by x
            sums_to_check_remove.add(2 * x)
            for z in current:
                if z != x:
                    sums_to_check_remove.add(x + z)
                    
            # Sums added by y (with S' = S \ {x})
            sums_to_check_add.add(2 * y)
            for z in current:
                if z != x:
                    sums_to_check_add.add(y + z)
            
            # Calculate delta
            # For s in sums_to_check_remove:
            #   if counts[s] == 3: delta -= 1
            #   counts[s] -= 1 (conceptually)
            
            # For s in sums_to_check_add:
            #   new_count = counts[s] - (1 if s in sums_to_check_remove else 0) + 1
            #   Wait, if s is in both, net change is 0.
            #   If s is only in remove: count decreases.
            #   If s is only in add: count increases.
            
            # Let's just compute the new count for each relevant sum.
            
            delta_conf = 0
            for s in sums_to_check_remove:
                if counts[s] == 3:
                    delta_conf -= 1
                # counts[s] effectively decreases by 1
            
            for s in sums_to_check_add:
                # Determine effective count before adding y
                # Base count
                c = counts[s]
                # If s was also removed, c decreased by 1
                if s in sums_to_check_remove:
                    c -= 1
                # Now add y
                # If c == 2, it becomes 3 (conflict +1)
                if c == 2:
                    delta_conf += 1
            
            # Acceptance
            if delta_conf < 0:
                # Accept
                # Perform swap
                current[x_idx] = y
                in_set[x] = False
                in_set[y] = True
                
                # Update counts
                for s in sums_to_check_remove:
                    counts[s] -= 1
                for s in sums_to_check_add:
                    counts[s] += 1
                    
                current_conflicts += delta_conf
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_solution = list(current)
                        if len(current) > best_size:
                            best_size = len(current)
            elif delta_conf == 0:
                # Accept swap (maintains size, might help future moves)
                current[x_idx] = y
                in_set[x] = False
                in_set[y] = True
                for s in sums_to_check_remove:
                    counts[s] -= 1
                for s in sums_to_check_add:
                    counts[s] += 1
            else:
                # Delta > 0
                if random.random() < math.exp(-delta_conf / T):
                    current[x_idx] = y
                    in_set[x] = False
                    in_set[y] = True
                    for s in sums_to_check_remove:
                        counts[s] -= 1
                    for s in sums_to_check_add:
                        counts[s] += 1
                    current_conflicts += delta_conf
            
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor

    # Final check: ensure best_solution is valid
    # Re-verify best_solution
    final_set = set(best_solution)
    final_counts = [0] * (max_sum + 1)
    for i in range(len(best_solution)):
        x = best_solution[i]
        final_counts[2 * x] += 1
        for j in range(i + 1, len(best_solution)):
            y = best_solution[j]
            final_counts[x + y] += 1
            
    valid = True
    for s in range(2, max_sum + 1):
        if final_counts[s] > 2:
            valid = False
            break
            
    if valid:
        return best_solution
    else:
        # Fallback to greedy if SA broke it (shouldn't happen if we track best_conflicts==0)
        return chosen

# EVOLVE-BLOCK-END
