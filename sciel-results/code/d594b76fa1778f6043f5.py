import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    fixed = [1, 2, 5, 11]
    
    # Initialize state
    current_set = fixed[:]
    sum_counts = {}
    
    def update_sums(x, sign):
        # sign 1 for add, -1 for remove
        for y in current_set:
            if x == y:
                s = 2*x
                if sign == 1:
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                else:
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
            else:
                s = x + y
                if sign == 1:
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                else:
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
                    
    # Initialize sums for fixed set
    # Note: update_sums assumes x is NOT in current_set for the 'add' logic to work simply?
    # No, if x is in current_set, we handle 2x.
    # But when adding, x is not in current_set.
    # When removing, x IS in current_set.
    
    # Let's just recompute sums initially
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            s = current_set[i] + current_set[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            
    best_set = current_set[:]
    best_size = len(current_set)
    
    temperature = 50.0
    cooling = 0.9995
    
    # Random seed for reproducibility in testing, but random for real run
    random.seed(42)
    
    while time.time() < deadline:
        # Move
        r = random.random()
        
        if r < 0.6:
            # Add
            if len(current_set) < 20:
                x = random.randint(1, 284)
                if x not in current_set:
                    # Check validity
                    valid = True
                    for y in current_set:
                        s1 = x + y
                        s2 = 2*x if y == x else None # y != x here
                        if sum_counts.get(s1, 0) > 0:
                            valid = False
                            break
                    if sum_counts.get(2*x, 0) > 0:
                        valid = False
                    
                    if valid:
                        # Add x
                        current_set.append(x)
                        for y in current_set[:-1]:
                            s = x + y
                            sum_counts[s] = sum_counts.get(s, 0) + 1
                        sum_counts[2*x] = sum_counts.get(2*x, 0) + 1
                        
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                    else:
                        # Reject or accept with penalty?
                        # Just reject for now to keep valid sets.
                        pass
        elif r < 0.8:
            # Remove
            if len(current_set) > 4:
                # Pick a non-fixed element
                candidates = [i for i in range(4, len(current_set))]
                if candidates:
                    idx = random.choice(candidates)
                    x = current_set.pop(idx)
                    # Remove sums
                    for y in current_set:
                        s = x + y
                        sum_counts[s] -= 1
                        if sum_counts[s] == 0:
                            del sum_counts[s]
                    sum_counts[2*x] -= 1
                    if sum_counts[2*x] == 0:
                        del sum_counts[2*x]
        else:
            # Swap
            if len(current_set) > 4:
                candidates = [i for i in range(4, len(current_set))]
                if candidates:
                    idx = random.choice(candidates)
                    x_old = current_set[idx]
                    x_new = random.randint(1, 284)
                    if x_new not in current_set:
                        # Check if swapping x_old with x_new is valid
                        # Remove x_old, add x_new
                        # This is complex to check without doing it.
                        # Let's just do Remove then Add in the next iterations?
                        # Or check validity of the new set.
                        
                        # Tentative set
                        temp_set = current_set[:]
                        temp_set[idx] = x_new
                        
                        # Check validity of temp_set
                        # This is O(N^2). N is small (~18). 18^2 = 324. Fast.
                        sums = set()
                        valid = True
                        for i in range(len(temp_set)):
                            for j in range(i, len(temp_set)):
                                s = temp_set[i] + temp_set[j]
                                if s in sums:
                                    valid = False
                                    break
                                sums.add(s)
                            if not valid: break
                        
                        if valid:
                            # Update state
                            # Remove x_old sums
                            for y in current_set:
                                if y != x_old:
                                    s = x_old + y
                                    sum_counts[s] -= 1
                                    if sum_counts[s] == 0:
                                        del sum_counts[s]
                            sum_counts[2*x_old] -= 1
                            if sum_counts[2*x_old] == 0:
                                del sum_counts[2*x_old]
                            
                            # Add x_new sums
                            for y in current_set:
                                if y != x_old:
                                    s = x_new + y
                                    sum_counts[s] = sum_counts.get(s, 0) + 1
                            sum_counts[2*x_new] = sum_counts.get(2*x_new, 0) + 1
                            
                            current_set[idx] = x_new
                            
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set[:]
            
        # Cooling
        temperature *= cooling
        
    return best_set
