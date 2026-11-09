import math
import random
import time

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds limit, 10s buffer
    
    # Precompute gcds or just use math.gcd
    # A number x is compatible with set S if lcm(x, s) <= 512 for all s in S
    
    def is_compatible(x, S):
        for s in S:
            if math.lcm(x, s) > 512:
                return False
        return True

    # Initial solution: 1..23
    # 23 is compatible with 1..22 because 23 is prime and 23*22 = 506 <= 512
    best_set = set(range(1, 24))
    best_size = len(best_set)
    
    # Greedy improvement: try to add any remaining number
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x not in best_set and is_compatible(x, best_set):
                best_set.add(x)
                changed = True
                if len(best_set) > best_size:
                    best_size = len(best_set)
                
    # Local Search: Simulated Annealing or Random Restart Hill Climbing
    # We will use a random restart hill climbing with swaps and additions
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Start from best known or a random valid set?
        # Starting from best known is usually better for exploitation.
        # Let's try starting from a random subset of the best set plus some random additions?
        # Or just perturb the best set.
        
        current_set = set(best_set)
        
        # Randomize order of elements to break symmetry?
        # Let's do a local search loop for a fixed number of steps or until time
        
        steps = 0
        while time.time() < deadline and steps < 10000:
            steps += 1
            if not current_set:
                break
                
            # Try to add a random element
            if random.random() < 0.5:
                # Pick a random candidate from [1, 512] not in set
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                if candidates:
                    x = random.choice(candidates)
                    if is_compatible(x, current_set):
                        current_set.add(x)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
            else:
                # Try a swap: remove one, add one
                # Remove a random element
                x_to_remove = random.choice(list(current_set))
                temp_set = set(current_set)
                temp_set.remove(x_to_remove)
                
                # Try to add a random element from candidates
                # To make it efficient, we can just pick a random candidate and check
                # If it fits, we swap.
                # But we should prefer candidates that are "likely" to fit.
                # Simply picking random from 1..512 might be slow if set is large.
                # Let's pick from a subset of "promising" candidates?
                # Or just random.
                
                # Optimization: Instead of checking all candidates, just try a few random ones
                found_swap = False
                for _ in range(10): # Try 10 random swaps
                    x_to_add = random.randint(1, N)
                    if x_to_add in temp_set:
                        continue
                    if is_compatible(x_to_add, temp_set):
                        # Swap successful
                        temp_set.add(x_to_add)
                        current_set = temp_set
                        found_swap = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break
                if not found_swap:
                    # Maybe just remove? No, we want to maximize size.
                    # If we can't add, we stay same (or revert)
                    # current_set remains unchanged in this branch if no swap found
                    # Actually, we modified temp_set but didn't assign to current_set if no swap.
                    # So current_set is unchanged.
                    pass

    # Final greedy pass on best_set
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x not in best_set and is_compatible(x, best_set):
                best_set.add(x)
                changed = True

    return list(best_set)
