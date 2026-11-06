"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def is_valid(arr):
        """
        Checks if arr is a valid Sidon set covering 1..30.
        Returns (is_valid, diffs_set, diameter).
        """
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None
            
        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)
        
        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def fast_check_sidon(arr):
        """Fast check for Sidon property only."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def generate_greedy_initial():
        """
        Greedy construction to get a valid starting set.
        We try to build a set that covers 1..30 with minimal diameter.
        """
        # Strategy: Randomized greedy with restarts
        best_set = None
        best_diam = float('inf')
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            current = [0]
            diffs = set()
            
            # We want to cover 1..30.
            # Greedily pick candidates that add new small differences.
            # Sort candidates by how many new small diffs they add?
            # Or just random order from 1..100.
            
            candidates = list(range(1, 100))
            random.shuffle(candidates)
            
            valid_so_far = True
            for c in candidates:
                # Check if adding c maintains Sidon
                new_diffs = set()
                ok = True
                for x in current:
                    d = abs(c - x)
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                # Add it
                current.append(c)
                diffs.update(new_diffs)
                current.sort()
                
                # Check if we've covered 1..30
                if all(d in diffs for d in range(1, 31)):
                    diam = max(current) - min(current)
                    if diam < best_diam:
                        best_diam = diam
                        best_set = current[:]
                    break # Stop this restart, we have a valid set
            
            # If we didn't break, we didn't cover 1..30. 
            # We could try to force coverage, but for now, if we have a valid set, use it.
            # If not, this restart failed.
            
            if best_set is not None and best_diam < 80:
                break

        if best_set is not None:
            return best_set
        
        # Fallback: Deterministic greedy
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                d = x - y
                if d in diffs:
                    valid = False
                    break
            if valid:
                for y in a:
                    diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    def local_search(initial_set):
        """
        Hill climbing / Simulated Annealing to minimize diameter.
        """
        current = list(initial_set)
        current.sort()
        valid, _, best_diam = is_valid(current)
        if not valid:
            return initial_set
        
        best_set = current[:]
        T = 50.0
        T_min = 0.5
        cooling = 0.9995
        
        while time.time() < deadline - 5:
            if T < T_min:
                T = T_min
            
            # Generate neighbor
            neighbor = current[:]
            move_type = random.choice(['swap', 'remove', 'add'])
            
            if move_type == 'swap':
                # Replace a non-zero element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    max_val = max(neighbor)
                    # Bias towards smaller values to reduce diameter
                    if random.random() < 0.7:
                        new_val = random.randint(1, max_val)
                    else:
                        new_val = random.randint(1, max_val + 20)
                    
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:] # No change
            
            elif move_type == 'remove' and len(neighbor) > 1:
                # Remove a non-zero element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    neighbor.pop(idx)
                    neighbor.sort()
            
            elif move_type == 'add' and len(neighbor) < 32:
                # Add a new element
                max_val = max(neighbor)
                # Try to fill gaps or extend
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val)
                else:
                    new_val = random.randint(1, max_val + 10)
                
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            
            # Evaluate
            if len(neighbor) > 32 or 0 not in neighbor:
                continue
            if max(neighbor) > 10**6:
                continue
            
            # Fast Sidon check first
            if not fast_check_sidon(neighbor):
                continue
            
            # Full validity check
            v, diffs, new_diam = is_valid(neighbor)
            if not v:
                continue
            
            delta = new_diam - best_diam
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current = neighbor
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_set = neighbor[:]
            
            T *= cooling
            
        return best_set

    # Main Loop
    best_overall = None
    best_overall_diam = float('inf')
    
    # Get initial set
    init_set = generate_greedy_initial()
    
    if init_set is not None:
        result = local_search(init_set)
        v, _, diam = is_valid(result)
        if v:
            best_overall = result
            best_overall_diam = diam
    
    # If we have time, try more restarts with different initializations?
    # The local search is quite strong. Let's just run it for the remaining time.
    # But we already ran it inside local_search until deadline - 5.
    # So we should just return best_overall.
    
    if best_overall is None:
        # Emergency fallback
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                d = x - y
                if d in diffs:
                    valid = False
                    break
            if valid:
                for y in a:
                    diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    return best_overall
# EVOLVE-BLOCK-END
