"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    N = 1000
    MAX_REP = 2
    
    def is_valid_addition(x, chosen, sum_counts):
        # Check if adding x to chosen keeps sum_counts <= MAX_REP
        # We need to check sums x+y for all y in chosen, and 2*x
        # We simulate the addition
        valid = True
        new_sums = []
        
        for y in chosen:
            s = x + y
            if sum_counts[s] + 1 > MAX_REP:
                return False
            new_sums.append(s)
        
        s_double = 2 * x
        if sum_counts[s_double] + 1 > MAX_REP:
            return False
        new_sums.append(s_double)
        
        return True

    def apply_addition(x, chosen, sum_counts):
        for y in chosen:
            sum_counts[x + y] += 1
        sum_counts[2 * x] += 1
        chosen.append(x)

    def apply_removal(x, chosen, sum_counts):
        chosen.remove(x)
        for y in chosen:
            sum_counts[x + y] -= 1
        sum_counts[2 * x] -= 1

    def greedy_init(seed):
        random.seed(seed)
        chosen = []
        sum_counts = [0] * (2 * N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, chosen, sum_counts):
                apply_addition(x, chosen, sum_counts)
        return chosen, sum_counts

    def local_search(chosen, sum_counts, steps=5000):
        # Hill Climbing / Simulated Annealing
        current_size = len(chosen)
        best_size = current_size
        best_chosen = list(chosen)
        
        # We will try swaps and additions/removals
        # To be fast, we only try valid moves
        # Start temperature high, decay
        temp = 1.0
        cooling = 0.999
        
        for step in range(steps):
            move_type = random.random()
            improved = False
            
            if move_type < 0.2:
                # Try to add a random element
                remaining = [x for x in range(1, N + 1) if x not in chosen]
                if remaining:
                    x = random.choice(remaining)
                    if is_valid_addition(x, chosen, sum_counts):
                        apply_addition(x, chosen, sum_counts)
                        if len(chosen) > best_size:
                            best_size = len(chosen)
                            best_chosen = list(chosen)
                        improved = True
            elif move_type < 0.5:
                # Try to remove a random element (to allow others)
                if chosen:
                    x = random.choice(chosen)
                    apply_removal(x, chosen, sum_counts)
                    # After removal, try to add random elements
                    added_any = False
                    remaining = [x for x in range(1, N + 1) if x not in chosen]
                    # Try a few additions
                    for _ in range(10):
                        if not remaining: break
                        y = random.choice(remaining)
                        if is_valid_addition(y, chosen, sum_counts):
                            apply_addition(y, chosen, sum_counts)
                            remaining.remove(y)
                            added_any = True
                            if len(chosen) > best_size:
                                best_size = len(chosen)
                                best_chosen = list(chosen)
                    # If we didn't gain, we are smaller. 
                    # In SA, we might accept worse states, but for simplicity, 
                    # let's only keep if we improved or if we want to explore.
                    # Actually, if we removed 1 and added 0, size decreased.
                    # We should probably revert if no improvement?
                    # Let's just track best.
            else:
                # Try a swap
                if chosen:
                    x = random.choice(chosen)
                    remaining = [y for y in range(1, N + 1) if y not in chosen]
                    if remaining:
                        y = random.choice(remaining)
                        # Check if swap is valid:
                        # Remove x, add y.
                        # We need to check validity of S \ {x} U {y}
                        # This is equivalent to checking if y can be added to S \ {x}
                        
                        # Temporarily remove x
                        chosen.remove(x)
                        for z in chosen:
                            sum_counts[x + z] -= 1
                        sum_counts[2 * x] -= 1
                        
                        if is_valid_addition(y, chosen, sum_counts):
                            apply_addition(y, chosen, sum_counts)
                            # Size is same. 
                            # But this might open up space for others?
                            # Try to add more
                            for _ in range(5):
                                rem = [z for z in range(1, N + 1) if z not in chosen]
                                if not rem: break
                                z = random.choice(rem)
                                if is_valid_addition(z, chosen, sum_counts):
                                    apply_addition(z, chosen, sum_counts)
                                    if len(chosen) > best_size:
                                        best_size = len(chosen)
                                        best_chosen = list(chosen)
                        else:
                            # Revert
                            apply_addition(x, chosen, sum_counts)
                            # Note: apply_addition assumes x is NOT in chosen. 
                            # We removed it. So this works.
                            # But wait, is_valid_addition checks against current sum_counts.
                            # We reverted sum_counts.
                            # So we just re-add x.
                            
            if not improved and random.random() < 0.1:
                # Random restart of local search? Or just accept worse state?
                # For now, we just keep the global best.
                pass
                
            temp *= cooling
            
        return best_chosen

    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Run a few random greeds to find a good starting point
    for seed in range(10):
        chosen, sum_counts = greedy_init(seed)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
            
    # Run local search on the best found so far
    # We need sum_counts for the best_set
    # Recreate sum_counts for best_set
    sum_counts = [0] * (2 * N + 1)
    for x in best_set:
        for y in best_set:
            if x <= y:
                sum_counts[x + y] += 1
        # Wait, the sum_counts logic in greedy_init counts x+y for all y in chosen (including x? No, y iterates chosen which doesn't include x yet).
        # In greedy_init, `for y in chosen` where x is not in chosen.
        # So it counts x+y for y in S. And 2x.
        # This covers all pairs (x,y) with x in S_new, y in S_old.
        # Since we add one by one, every pair is counted exactly once when the second element is added?
        # If we add a, then b.
        # Add a: counts 2a.
        # Add b: counts b+a, 2b.
        # Total: 2a, 2b, a+b. Correct.
        
    # Re-verify sum_counts for best_set to be safe
    sum_counts = [0] * (2 * N + 1)
    for i in range(len(best_set)):
        for j in range(i, len(best_set)):
            sum_counts[best_set[i] + best_set[j]] += 1

    # Run local search
    improved_set = local_search(best_set, sum_counts, steps=20000)
    
    if len(improved_set) > best_size:
        best_set = improved_set
        best_size = len(best_set)
        
    # If time remains, try more seeds
    while time.time() - start_time < 110:
        seed = random.randint(0, 10000)
        chosen, sum_counts = greedy_init(seed)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
            # Re-verify sum_counts for this new best
            sum_counts = [0] * (2 * N + 1)
            for i in range(len(best_set)):
                for j in range(i, len(best_set)):
                    sum_counts[best_set[i] + best_set[j]] += 1
            # Try a short local search
            improved_set = local_search(best_set, sum_counts, steps=5000)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END
