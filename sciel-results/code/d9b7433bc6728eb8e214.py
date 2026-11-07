import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    
    def is_valid(s_set):
        """Check if no three distinct a,b,c in s_set have all pair sums in s_set."""
        elems = sorted(s_set)
        n = len(elems)
        in_set = set(elems)
        
        # Optimization: If a+b is in set, then a+b <= 128.
        # Also, for a fixed a,b, we only need to check c > b.
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                s_ab = a + b
                if s_ab > N or s_ab not in in_set:
                    continue
                # If s_ab is in set, check for c
                # c must be > b
                for k in range(j + 1, n):
                    c = elems[k]
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac in in_set and s_bc in in_set:
                        return False
        return True

    # Initial candidate: upper half
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    random.seed(42)
    
    # Local search function
    def local_search(current_set, max_steps=1000):
        current = set(current_set)
        steps = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            improved = False
            
            # Try adding elements
            candidates = [x for x in range(1, N + 1) if x not in current]
            random.shuffle(candidates)
            
            for elem in candidates:
                trial = current | {elem}
                if is_valid(trial):
                    current = trial
                    improved = True
                    break
            
            if improved:
                continue
                
            # If no single addition works, try swaps (remove 1, add 1) to rearrange
            # Or remove 1, add 2
            # Let's try removing one element to see if we can add more
            # Heuristic: remove an element that is in many "near-miss" triples?
            # Simpler: random removal, then try to add 2 elements
            
            if len(current) > 0:
                # Pick a random element to remove
                to_remove = random.choice(list(current))
                trial_removed = current - {to_remove}
                
                # Try to add 2 elements from candidates
                candidates = [x for x in range(1, N + 1) if x not in trial_removed]
                random.shuffle(candidates)
                
                added = 0
                new_current = set(trial_removed)
                for elem in candidates:
                    if added == 2:
                        break
                    trial_add = new_current | {elem}
                    if is_valid(trial_add):
                        new_current = trial_add
                        added += 1
                
                if len(new_current) > len(current):
                    current = new_current
                    improved = True
        
        return current

    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Different initial sets
        if restart == 0:
            init = set(range(N // 2 + 1, N + 1))
        elif restart == 1:
            # Try a mix
            init = set(range(40, N + 1))
            # Ensure validity
            while not is_valid(init):
                # Remove random element
                to_remove = random.choice(list(init))
                init.remove(to_remove)
        else:
            # Random subset of upper half + some lower
            init = set()
            for x in range(N // 2 + 1, N + 1):
                if random.random() < 0.8:
                    init.add(x)
            for x in range(1, N // 2 + 1):
                if random.random() < 0.3:
                    init.add(x)
            
            # Ensure validity
            while not is_valid(init):
                if len(init) == 0:
                    break
                to_remove = random.choice(list(init))
                init.remove(to_remove)
        
        # Run local search
        improved_set = local_search(init)
        
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final polish: try to add any remaining element
    remaining = set(range(1, N + 1)) - best_set
    for elem in remaining:
        if time.time() > deadline - 2:
            break
        trial = best_set | {elem}
        if is_valid(trial):
            best_set = trial
    
    # Ensure return type is correct
    result = list(best_set)
    result = [int(x) for x in result]
    return result
