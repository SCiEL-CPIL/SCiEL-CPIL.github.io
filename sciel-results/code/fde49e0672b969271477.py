import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # 1. Greedy Initial Solution
    # We want to maximize size. A good heuristic is to pick numbers that are "sparse" in terms of sum collisions.
    # Simple greedy: iterate 1..N, add if valid.
    # Better greedy: Randomized greedy with restarts to find a good initial state.
    
    def greedy_init():
        chosen = []
        counts = [0] * (max_sum + 1)
        
        # Randomize the order to get diverse initial solutions
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            # Check 2x
            if counts[2 * x] >= 2:
                valid = False
            else:
                # Check x + y for y in chosen
                for y in chosen:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
                    
        return chosen, counts

    best_set = []
    best_size = 0
    best_counts = [0] * (max_sum + 1)
    
    # Try a few greedy initializations to find a good starting point
    for _ in range(10):
        if time.time() > deadline - 10: break
        init_set, init_counts = greedy_init()
        size = len(init_set)
        if size > best_size:
            best_size = size
            best_set = init_set
            best_counts = init_counts

    # 2. Iterated Local Search with Simulated Annealing
    
    # State variables
    current_set = best_set.copy()
    in_set = [False] * (N + 1)
    for x in current_set:
        in_set[x] = True
        
    counts = best_counts.copy()
    
    # Precompute list of available elements
    available = [x for x in range(1, N + 1) if not in_set[x]]
    
    def get_conflicts(counts):
        c = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                c += counts[s] - 2
        return c

    current_conflicts = get_conflicts(counts)
    best_conflicts = current_conflicts
    best_solution = current_set.copy()
    best_solution_counts = counts.copy()
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    cooling = 0.999
    
    def calc_delta_swap(x, y, current, counts, in_set):
        """
        Calculate delta conflicts for swapping x (out) with y (in).
        x is in current, y is not.
        """
        # Sums affected by removing x
        # 2x
        # x + z for z in current, z != x
        
        # Sums affected by adding y
        # 2y
        # y + z for z in current, z != x
        
        # We can just iterate through current and check affected sums.
        # To be efficient, we track the change in conflict for each sum.
        
        delta = 0
        
        # Helper to update delta for a sum s changing by +1 or -1
        # We need old count to calculate old conflict, new count for new conflict.
        
        # Sums to decrement
        dec_sums = set()
        dec_sums.add(2 * x)
        for z in current:
            if z != x:
                dec_sums.add(x + z)
                
        # Sums to increment
        inc_sums = set()
        inc_sums.add(2 * y)
        for z in current:
            if z != x:
                inc_sums.add(y + z)
                
        all_sums = dec_sums.union(inc_sums)
        
        for s in all_sums:
            old_c = counts[s]
            new_c = old_c
            if s in dec_sums:
                new_c -= 1
            if s in inc_sums:
                new_c += 1
                
            old_conf = max(0, old_c - 2)
            new_conf = max(0, new_c - 2)
            delta += new_conf - old_conf
            
        return delta, dec_sums, inc_sums

    def calc_delta_add(x, current, counts):
        """
        Calculate delta conflicts for adding x.
        """
        delta = 0
        inc_sums = set()
        inc_sums.add(2 * x)
        for z in current:
            inc_sums.add(x + z)
            
        for s in inc_sums:
            old_c = counts[s]
            new_c = old_c + 1
            old_conf = max(0, old_c - 2)
            new_conf = max(0, new_c - 2)
            delta += new_conf - old_conf
            
        return delta, inc_sums

    def calc_delta_remove(x, current, counts):
        """
        Calculate delta conflicts for removing x.
        """
        delta = 0
        dec_sums = set()
        dec_sums.add(2 * x)
        for z in current:
            if z != x:
                dec_sums.add(x + z)
                
        for s in dec_sums:
            old_c = counts[s]
            new_c = old_c - 1
            old_conf = max(0, old_c - 2)
            new_conf = max(0, new_c - 2)
            delta += new_conf - old_conf
            
        return delta, dec_sums

    # Main ILS Loop
    no_improve_steps = 0
    max_no_improve = 5000 # Steps before forcing perturbation
    
    while time.time() < deadline:
        T = T_start
        local_no_improve = 0
        max_local_no_improve = 2000
        
        while time.time() < deadline and T > T_end:
            move_type = random.random()
            
            if move_type < 0.4:
                # Swap
                if not current_set or not available:
                    T *= cooling
                    continue
                x_idx = random.randint(0, len(current_set) - 1)
                x = current_set[x_idx]
                y_idx = random.randint(0, len(available) - 1)
                y = available[y_idx]
                
                delta, dec_sums, inc_sums = calc_delta_swap(x, y, current_set, counts, in_set)
                
                # Acceptance criterion
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
                if accept:
                    # Apply swap
                    in_set[x] = False
                    in_set[y] = True
                    current_set[x_idx] = y
                    available[y_idx] = x
                    
                    for s in dec_sums:
                        counts[s] -= 1
                    for s in inc_sums:
                        counts[s] += 1
                        
                    current_conflicts += delta
                    no_improve_steps = 0
                    local_no_improve = 0
                    
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_solution = current_set.copy()
                        best_solution_counts = counts.copy()
                        # Update global best size if conflicts are 0
                        if best_conflicts == 0:
                            if len(current_set) > best_size:
                                best_size = len(current_set)
            elif move_type < 0.7:
                # Add
                if not available:
                    T *= cooling
                    continue
                y_idx = random.randint(0, len(available) - 1)
                y = available[y_idx]
                
                delta, inc_sums = calc_delta_add(y, current_set, counts)
                
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
                if accept:
                    # Apply add
                    in_set[y] = True
                    current_set.append(y)
                    available.pop(y_idx)
                    
                    for s in inc_sums:
                        counts[s] += 1
                        
                    current_conflicts += delta
                    no_improve_steps = 0
                    local_no_improve = 0
                    
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_solution = current_set.copy()
                        best_solution_counts = counts.copy()
                        if best_conflicts == 0:
                            if len(current_set) > best_size:
                                best_size = len(current_set)
            else:
                # Remove
                if not current_set:
                    T *= cooling
                    continue
                x_idx = random.randint(0, len(current_set) - 1)
                x = current_set[x_idx]
                
                delta, dec_sums = calc_delta_remove(x, current_set, counts)
                
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    accept = random.random() < math.exp(-delta / T)
                    
                if accept:
                    # Apply remove
                    in_set[x] = False
                    current_set.pop(x_idx)
                    available.append(x)
                    
                    for s in dec_sums:
                        counts[s] -= 1
                        
                    current_conflicts += delta
                    no_improve_steps = 0
                    local_no_improve = 0
                    
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_solution = current_set.copy()
                        best_solution_counts = counts.copy()
                        if best_conflicts == 0:
                            if len(current_set) > best_size:
                                best_size = len(current_set)
            
            T *= cooling
            local_no_improve += 1
            
            if local_no_improve > max_local_no_improve:
                break

        # Perturbation if stuck
        if no_improve_steps > max_no_improve or local_no_improve > max_local_no_improve:
            # Perform a strong perturbation
            # Remove 10-20 random elements
            num_remove = min(len(current_set), random.randint(10, 20))
            for _ in range(num_remove):
                if not current_set: break
                x_idx = random.randint(0, len(current_set) - 1)
                x = current_set[x_idx]
                
                # Calculate delta for removal
                delta, dec_sums = calc_delta_remove(x, current_set, counts)
                
                # Apply removal unconditionally
                in_set[x] = False
                current_set.pop(x_idx)
                available.append(x)
                
                for s in dec_sums:
                    counts[s] -= 1
                    
                current_conflicts += delta
                
            # Re-add some elements greedily or randomly to restore size
            # Let's try to add back random available elements if they don't increase conflicts too much
            # Or just restart SA from this perturbed state
            
            no_improve_steps = 0
            local_no_improve = 0
            T = T_start * 0.5 # Lower start temp for perturbation recovery? Or higher?
            # Actually, higher temp helps escape. Let's restart with T_start.
            T = T_start

    # Final Check and Return
    # We want to return a valid set (conflicts == 0) with maximum size.
    # If best_conflicts > 0, we must repair or fall back to a known valid set.
    # The problem requires exact count <= 2.
    
    # If best_solution has conflicts, we need to make it valid.
    # Simplest: if best_conflicts > 0, we might need to remove elements until valid.
    # But we track best_solution only when conflicts decreased. 
    # If we started with a valid set (conflicts=0), best_conflicts should be 0.
    # Greedy init produces a valid set. So best_conflicts should be 0 initially.
    # SA only updates best_solution if current_conflicts < best_conflicts.
    # So if best_conflicts was 0, it stays 0.
    # Thus best_solution is always valid.
    
    # However, we might have found a larger set with conflicts > 0 that we didn't save as best_solution.
    # We only save if it's better (lower conflicts).
    # So best_solution is the best valid set found (or the initial valid set if no better valid set found).
    # Wait, if we find a set with conflicts=1 and size=100, and best is conflicts=0 size=90.
    # We do NOT update best_solution because 1 > 0.
    # So best_solution remains the valid set of size 90.
    # This is correct behavior for the task (must be valid).
    
    # What if we can repair the larger set?
    # If we have a set with size 100 and conflicts 1, we can remove one element to make it valid.
    # Size becomes 99, which is better than 90.
    
    # Let's add a final polish step:
    # If best_conflicts > 0, try to repair best_solution.
    # But we don't store the "largest invalid" set.
    # Let's modify the logic: store best_valid_size and best_valid_set.
    # And also track current set. At the end, if current set is invalid, try to repair it.
    
    # Actually, the current logic keeps best_solution as the best VALID set found.
    # This is safe.
    
    # Let's verify best_solution is valid.
    final_set = best_solution
    final_counts = [0] * (max_sum + 1)
    for i in range(len(final_set)):
        x = final_set[i]
        final_counts[2 * x] += 1
        for j in range(i + 1, len(final_set)):
            y = final_set[j]
            final_counts[x + y] += 1
            
    valid = True
    for s in range(2, max_sum + 1):
        if final_counts[s] > 2:
            valid = False
            break
            
    if not valid:
        # Fallback to greedy init if something went wrong
        # This shouldn't happen if best_conflicts was tracked correctly from a valid start
        return best_set 
        
    return final_set

