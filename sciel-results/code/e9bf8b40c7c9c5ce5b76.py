"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0  # Safety margin before 90s kill
    N = 4096
    
    def compute_triple_sums(selected):
        """Compute all triple sums for a given selected set."""
        sums = set()
        k = len(selected)
        for i in range(k):
            a = selected[i]
            for j in range(i, k):
                b = selected[j]
                for l in range(j, k):
                    c = selected[l]
                    sums.add(a + b + c)
        return sums
    
    def can_add(x, selected, triple_sums):
        """
        Check if x can be added to selected without creating duplicate triple sums.
        Returns the set of new triple sums if valid, else None.
        """
        new_sums = set()
        
        # 1. x + x + x
        s = 3 * x
        if s in triple_sums:
            return None
        new_sums.add(s)
        
        # 2. x + x + y for y in selected
        for y in selected:
            s = 2 * x + y
            if s in triple_sums or s in new_sums:
                return None
            new_sums.add(s)
            
        # 3. x + y + z for y, z in selected (y <= z)
        k = len(selected)
        for i in range(k):
            y = selected[i]
            s = x + 2 * y
            if s in triple_sums or s in new_sums:
                return None
            new_sums.add(s)
            
            for j in range(i + 1, k):
                z = selected[j]
                s = x + y + z
                if s in triple_sums or s in new_sums:
                    return None
                new_sums.add(s)
                
        return new_sums
    
    def greedy_build_smallest_first(target_size):
        """Greedy: pick smallest valid x each time."""
        selected = []
        triple_sums = set()
        
        for x in range(1, N + 1):
            if len(selected) >= target_size:
                break
            if time.time() > DEADLINE:
                break
            new_sums = can_add(x, selected, triple_sums)
            if new_sums is not None:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected
    
    def greedy_build_random(target_size, seed_offset=0):
        """Greedy with random candidate selection."""
        rng = random.Random(seed_offset)
        selected = []
        triple_sums = set()
        selected_set = set()
        
        for _ in range(target_size):
            if time.time() > DEADLINE:
                break
            added = False
            # Try multiple random candidates
            for _ in range(200):
                x = rng.randint(1, N)
                if x in selected_set:
                    continue
                new_sums = can_add(x, selected, triple_sums)
                if new_sums is not None:
                    selected.append(x)
                    selected_set.add(x)
                    triple_sums.update(new_sums)
                    added = True
                    break
            if not added:
                break
        
        return selected
    
    def greedy_build_weighted(target_size, seed_offset=0):
        """Greedy with weighted random selection favoring smaller numbers."""
        rng = random.Random(seed_offset)
        selected = []
        triple_sums = set()
        selected_set = set()
        
        for _ in range(target_size):
            if time.time() > DEADLINE:
                break
            added = False
            for _ in range(300):
                # Weight towards smaller numbers
                x = int(rng.expovariate(1.0 / 500)) + 1
                if x > N:
                    x = rng.randint(1, N)
                if x in selected_set:
                    continue
                new_sums = can_add(x, selected, triple_sums)
                if new_sums is not None:
                    selected.append(x)
                    selected_set.add(x)
                    triple_sums.update(new_sums)
                    added = True
                    break
            if not added:
                break
        
        return selected
    
    def local_search_improve(initial_set, max_iterations=50):
        """Try to improve by removing one element and adding two new ones."""
        selected = list(initial_set)
        triple_sums = compute_triple_sums(selected)
        best_set = selected[:]
        best_size = len(selected)
        
        for iteration in range(max_iterations):
            if time.time() > DEADLINE:
                break
            
            improved = False
            # Try removing each element
            for i in range(len(selected)):
                if time.time() > DEADLINE:
                    return best_set
                
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                remaining_set = set(remaining)
                
                # Compute triple sums of remaining
                rem_sums = set()
                k = len(remaining)
                for a_idx in range(k):
                    a = remaining[a_idx]
                    for b_idx in range(a_idx, k):
                        b = remaining[b_idx]
                        for c_idx in range(b_idx, k):
                            c = remaining[c_idx]
                            rem_sums.add(a + b + c)
                
                # Try adding two new elements
                found_pair = False
                for x in range(1, N + 1):
                    if found_pair:
                        break
                    if x in remaining_set:
                        continue
                    
                    # Check if x can be added to remaining
                    new_sums_x = set()
                    valid_x = True
                    s = 3 * x
                    if s in rem_sums:
                        continue
                    new_sums_x.add(s)
                    for y in remaining:
                        s = 2 * x + y
                        if s in rem_sums or s in new_sums_x:
                            valid_x = False
                            break
                        new_sums_x.add(s)
                    if not valid_x:
                        continue
                    for ii in range(k):
                        y = remaining[ii]
                        s = x + 2 * y
                        if s in rem_sums or s in new_sums_x:
                            valid_x = False
                            break
                        new_sums_x.add(s)
                        if not valid_x:
                            break
                        for jj in range(ii + 1, k):
                            z = remaining[jj]
                            s = x + y + z
                            if s in rem_sums or s in new_sums_x:
                                valid_x = False
                                break
                            new_sums_x.add(s)
                            if not valid_x:
                                break
                        if not valid_x:
                            break
                    if not valid_x:
                        continue
                    
                    # Now try to add a second element y
                    for y in range(1, N + 1):
                        if y in remaining_set or y == x:
                            continue
                        
                        # Check if y can be added to remaining + {x}
                        new_sums_y = set()
                        valid_y = True
                        s = 3 * y
                        if s in rem_sums or s in new_sums_x:
                            continue
                        new_sums_y.add(s)
                        
                        # Check against remaining
                        for z in remaining:
                            s = 2 * y + z
                            if s in rem_sums or s in new_sums_x or s in new_sums_y:
                                valid_y = False
                                break
                            new_sums_y.add(s)
                        if not valid_y:
                            continue
                        for ii in range(k):
                            z = remaining[ii]
                            s = y + 2 * z
                            if s in rem_sums or s in new_sums_x or s in new_sums_y:
                                valid_y = False
                                break
                            new_sums_y.add(s)
                            if not valid_y:
                                break
                            for jj in range(ii + 1, k):
                                w = remaining[jj]
                                s = y + z + w
                                if s in rem_sums or s in new_sums_x or s in new_sums_y:
                                    valid_y = False
                                    break
                                new_sums_y.add(s)
                                if not valid_y:
                                    break
                            if not valid_y:
                                break
                        if not valid_y:
                            continue
                        
                        # Check x + x + y, x + y + y, x + y + x (already covered)
                        # x + x + y
                        s = 2 * x + y
                        if s in rem_sums or s in new_sums_x or s in new_sums_y:
                            continue
                        new_sums_y.add(s)
                        
                        # x + y + y  
                        s = x + 2 * y
                        if s in rem_sums or s in new_sums_x or s in new_sums_y:
                            continue
                        new_sums_y.add(s)
                        
                        # x + x + x already in new_sums_x
                        # x + y + z where z is from remaining - already checked above
                        
                        # All checks passed!
                        new_selected = remaining + [x, y]
                        if len(new_selected) > best_size:
                            best_set = new_selected[:]
                            best_size = len(new_selected)
                            improved = True
                            found_pair = True
                            break
                        # Even if same size, we might want to continue for better structure
                        # But for now, only accept if strictly better
                        break  # Only try one y per x for speed
                
                if found_pair:
                    break
            
            if not improved:
                break
        
        return best_set
    
    # Strategy: Try multiple approaches and keep the best
    best_set = []
    
    # 1. Deterministic greedy (smallest first) - fast baseline
    current = greedy_build_smallest_first(50)
    if len(current) > len(best_set):
        best_set = current[:]
    
    # 2. Try random greedy with different seeds
    for seed in range(10):
        if time.time() > DEADLINE - 10:
            break
        current = greedy_build_random(50, seed_offset=seed * 1000)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # 3. Try weighted greedy
    for seed in range(5):
        if time.time() > DEADLINE - 10:
            break
        current = greedy_build_weighted(50, seed_offset=seed * 2000 + 500)
        if len(current) > len(best_set):
            best_set = current[:]
    
    # 4. Local search improvement on the best set found
    if len(best_set) > 0 and time.time() < DEADLINE - 5:
        improved = local_search_improve(best_set, max_iterations=20)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set

# EVOLVE-BLOCK-END
