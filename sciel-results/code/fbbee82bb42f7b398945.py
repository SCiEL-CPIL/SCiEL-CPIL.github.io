import random
import time

# EVOLVE-BLOCK-START
def construct():
    MAX_SUM = 2000
    N = 1000
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer

    def greedy_construct(order):
        counts = [0] * (MAX_SUM + 1)
        chosen = []
        chosen_set = set()
        in_set = [False] * (N + 1)
        
        for x in order:
            valid = True
            # Check against existing elements
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            
            if valid:
                # Check self-sum
                if counts[2 * x] >= 2:
                    valid = False
            
            if valid:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_set.add(x)
                in_set[x] = True
        
        return chosen, chosen_set, counts, in_set

    best_size = 0
    best_chosen = []
    best_counts = []
    best_set = set()
    best_in_set = [False] * (N + 1)
    
    candidates = list(range(1, N + 1))
    
    # 1. Find a good initial solution via multiple greedy restarts
    # Mix of random and structured orders
    for i in range(100):
        if i < 10:
            order = candidates[:]
            if i % 2 == 0:
                order.sort()
            else:
                order.sort(reverse=True)
        else:
            order = candidates[:]
            random.shuffle(order)
            
        chosen, chosen_set, counts, in_set = greedy_construct(order)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen
            best_counts = counts
            best_set = chosen_set
            best_in_set = in_set

    # 2. Local Search
    current_chosen = best_chosen[:]
    current_set = best_set.copy()
    current_counts = best_counts[:]
    current_in_set = best_in_set[:]
    
    # Build excluded list
    excluded = [x for x in range(1, N + 1) if not current_in_set[x]]
    
    steps = 0
    last_improvement_time = start_time
    
    while time.time() - start_time < time_limit:
        if not excluded:
            break
        
        # Try to add an element
        added = False
        # Randomly sample a few candidates to add
        # If we iterate through all, it's O(N*K). N=1000, K~700. 700k ops.
        # Doing this every step might be slow if we do many steps.
        # Let's try a random sample first.
        
        # Pick a random excluded element
        c = random.choice(excluded)
        
        # Check if c can be added
        valid = True
        for y in current_chosen:
            if current_counts[c + y] >= 2:
                valid = False
                break
        if valid and current_counts[2 * c] >= 2:
            valid = False
            
        if valid:
            for y in current_chosen:
                current_counts[c + y] += 1
            current_counts[2 * c] += 1
            current_chosen.append(c)
            current_set.add(c)
            current_in_set[c] = True
            excluded.remove(c)
            if len(current_chosen) > best_size:
                best_size = len(current_chosen)
                best_chosen = current_chosen[:]
                best_counts = current_counts[:]
                best_set = current_set.copy()
                best_in_set = current_in_set[:]
                last_improvement_time = time.time()
            added = True
        else:
            # Try a swap: remove random r, add c
            # We only do this if we are stuck for a while or randomly
            if random.random() < 0.3 and current_chosen:
                idx = random.randint(0, len(current_chosen) - 1)
                r = current_chosen[idx]
                
                # Temporarily remove r
                for y in current_chosen:
                    if y != r:
                        current_counts[r + y] -= 1
                current_counts[2 * r] -= 1
                # Save r's position to restore if needed
                # Actually, let's just swap in memory carefully
                
                # Try to add c
                valid_c = True
                for y in current_chosen:
                    if y != r: # r is still in list, but counts are updated? 
                        # Wait, the loop above subtracted counts for r+y where y!=r.
                        # The list still contains r. We should remove r from list first for clarity?
                        # No, the check `for y in current_chosen` will see r.
                        # But counts for r+r are subtracted.
                        # If we check c+r, counts[c+r] is unchanged by removing r?
                        # Removing r subtracts count[r+y]. It does NOT change count[c+y] unless y=r.
                        # If y=r, count[c+r] is decreased by 1.
                        # So we must account for this.
                        
                        pass # Logic is getting messy. Let's do a cleaner swap.
                
                # Cleaner Swap Logic:
                # 1. Remove r from counts and list.
                # 2. Check if c is valid against new state.
                # 3. If valid, add c.
                # 4. If not, restore r.
                
                # Let's restart the swap logic properly.
                # We already subtracted counts for r. 
                # But the list `current_chosen` still has r.
                # Let's pop r.
                current_chosen.pop(idx)
                
                valid_c = True
                for y in current_chosen:
                    if current_counts[c + y] >= 2:
                        valid_c = False
                        break
                if valid_c and current_counts[2 * c] >= 2:
                    valid_c = False
                    
                if valid_c:
                    for y in current_chosen:
                        current_counts[c + y] += 1
                    current_counts[2 * c] += 1
                    current_chosen.append(c)
                    current_set.remove(r)
                    current_set.add(c)
                    current_in_set[r] = False
                    current_in_set[c] = True
                    # Update excluded
                    if r not in excluded:
                        excluded.append(r)
                    if c in excluded:
                        excluded.remove(c)
                else:
                    # Restore r
                    for y in current_chosen:
                        current_counts[r + y] += 1
                    current_counts[2 * r] += 1
                    current_chosen.append(r)
                    current_set.add(r)
                    current_in_set[r] = True

        steps += 1

    return best_chosen
# EVOLVE-BLOCK-END
