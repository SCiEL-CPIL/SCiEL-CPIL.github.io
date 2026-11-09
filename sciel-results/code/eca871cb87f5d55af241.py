import time
import random

def construct():
    L = 232792560
    weights = {i: L // i for i in range(1, 23)}
    
    def is_valid(S):
        if not S:
            return True
        sums = {0}
        for w in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + weights[w])
            sums.update(new_sums)
            # Early exit if collision detected
            # The number of sums should be 2**k after k elements
            # We can check len(sums) < 2**k
            # But we don't track k explicitly in this loop.
            # Just checking at the end is safer and fast enough for N<=22.
        return len(sums) == (1 << len(S))

    def greedy_init():
        S = set()
        # Randomize order for diversity
        elems = list(range(1, 23))
        random.shuffle(elems)
        for i in elems:
            if is_valid(S | {i}):
                S.add(i)
        return S

    start_time = time.time()
    deadline = start_time + 85
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local Search
    current_set = set(best_set)
    
    # To avoid getting stuck in a specific local minimum, we use a "temperature" or just random restarts.
    # We'll run multiple restarts within the time budget.
    
    for restart in range(50): # Max 50 restarts, but break on time
        if time.time() > deadline:
            break
            
        # If we found a very large set, we might want to focus on improving it.
        # But restarts help find larger sets.
        
        # Initialize current_set
        # Sometimes start from best_set, sometimes from new greedy
        if restart == 0:
            current_set = set(best_set)
        else:
            # Mix: start from best_set but perturb? Or new greedy?
            # New greedy is safer to find different basins.
            current_set = greedy_init()
            
        # Local search phase
        stuck_count = 0
        while time.time() < deadline:
            # Try to improve
            improved = False
            
            # 1. Try to Add
            complement = [i for i in range(1, 23) if i not in current_set]
            if complement:
                # Try a few random adds
                for _ in range(5):
                    elem = random.choice(complement)
                    new_set = current_set | {elem}
                    if is_valid(new_set):
                        current_set = new_set
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break
                if improved:
                    stuck_count = 0
                    continue
            
            # 2. If no add, try Swap
            # Pick a random element to remove, and a random element to add
            if current_set:
                swap_found = False
                for _ in range(10):
                    rem = random.choice(list(current_set))
                    comp = [i for i in range(1, 23) if i not in current_set]
                    if not comp:
                        break
                    add = random.choice(comp)
                    if rem == add:
                        continue
                    new_set = (current_set - {rem}) | {add}
                    if is_valid(new_set):
                        current_set = new_set
                        swap_found = True
                        break
                if swap_found:
                    stuck_count = 0
                    continue
            
            # 3. If no swap, try Remove (to escape)
            # Only remove if we are stuck and want to try a different configuration
            if current_set and stuck_count > 5:
                rem = random.choice(list(current_set))
                new_set = current_set - {rem}
                if is_valid(new_set):
                    current_set = new_set
                    stuck_count = 0
                    continue
            
            # If nothing worked, we are stuck
            stuck_count += 1
            if stuck_count > 20:
                break # Break inner loop, restart

    return list(best_set)
