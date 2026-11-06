import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    base_elements = [1, 2, 5, 11]
    base_set = set(base_elements)
    
    def compute_energy(lst):
        """
        Compute the number of duplicate differences in the list.
        Returns 0 if it's a valid Sidon set (Golomb ruler).
        """
        diffs = []
        n = len(lst)
        for i in range(n):
            ai = lst[i]
            for j in range(i + 1, n):
                diffs.append(abs(ai - lst[j]))
        
        # Count duplicates
        count = 0
        seen = set()
        for d in diffs:
            if d in seen:
                count += 1
            else:
                seen.add(d)
        return count

    def solve_for_size(k):
        """
        Try to find a valid Sidon set of size k using Simulated Annealing.
        Returns the list if found, else None.
        """
        if k < 4:
            return None
        
        # Initialize
        # We need k elements. 4 are fixed. k-4 are random.
        available = [x for x in range(1, 285) if x not in base_set]
        random.shuffle(available)
        
        # Ensure we have enough available elements
        if len(available) < k - 4:
            return None
            
        current_lst = base_elements + available[:k-4]
        random.shuffle(current_lst)
        
        current_energy = compute_energy(current_lst)
        
        if current_energy == 0:
            return sorted(current_lst)
            
        best_lst = sorted(current_lst[:])
        best_energy = current_energy
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        max_iterations = 500000 # Adjust based on time
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Pick two indices to modify.
            # One option: swap two non-base elements.
            # Another: replace one non-base element with a random one.
            
            # Let's stick to swapping and replacing non-base elements to keep structure somewhat stable
            non_base_indices = [i for i, x in enumerate(current_lst) if x not in base_set]
            
            if not non_base_indices:
                # Only base elements, can't move
                break
            
            move_type = random.random()
            
            if move_type < 0.5 and len(non_base_indices) >= 2:
                # Swap two non-base elements
                i, j = random.sample(non_base_indices, 2)
                new_lst = current_lst[:]
                new_lst[i], new_lst[j] = new_lst[j], new_lst[i]
            else:
                # Replace one non-base element with a random one from available
                i = random.choice(non_base_indices)
                # Pick a random candidate from available that is not in current_lst
                # To avoid duplicates, we can just pick from available and check
                # Or better, pick from all 1..284 and ensure uniqueness
                new_val = random.randint(1, 284)
                # Ensure new_val is not already in current_lst
                if new_val in current_lst:
                    # Retry or pick another
                    # For simplicity, if collision, skip this move or pick another
                    # Let's just pick another random one
                    attempts = 0
                    while new_val in current_lst and attempts < 10:
                        new_val = random.randint(1, 284)
                        attempts += 1
                    if new_val in current_lst:
                        continue
                
                new_lst = current_lst[:]
                new_lst[i] = new_val
            
            new_energy = compute_energy(new_lst)
            
            # Acceptance
            delta_E = new_energy - current_energy
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                current_lst = new_lst
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_lst = sorted(current_lst[:])
                    
                    if best_energy == 0:
                        return best_lst
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        if best_energy == 0:
            return best_lst
        return None

    # Try sizes from 20 down to 15
    for k in range(20, 14, -1):
        if time.time() > deadline - 5:
            break
            
        # Run multiple attempts for each size if time permits
        attempts = 0
        max_attempts = 10
        
        while time.time() < deadline - 5:
            sol = solve_for_size(k)
            if sol:
                return sol
            attempts += 1
            if attempts >= max_attempts:
                break
            
    # If we couldn't find a perfect solution, return the best partial one?
    # The problem requires a valid Sidon set.
    # If we fail to find one of size > 4, we should at least return the base set which is valid.
    # Let's verify base set is valid.
    # 1+2=3, 1+5=6, 1+11=12, 2+5=7, 2+11=13, 5+11=16.
    # Sums: 3,6,12,7,13,16. All distinct. Yes.
    
    # As a last resort, try a simple greedy to see if we can get anything better than base
    # But the prompt implies we should maximize. If SA fails, base is the safest valid answer.
    
    # Let's add a quick greedy fallback just in case
    def greedy_construct():
        lst = list(base_elements)
        sums = set()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
                
        for x in range(12, 285):
            if x in lst:
                continue
            valid = True
            for y in lst:
                if x + y in sums:
                    valid = False
                    break
            if valid:
                if 2*x in sums:
                    valid = False
            
            if valid:
                lst.append(x)
                for y in lst[:-1]:
                    sums.add(x + y)
                sums.add(2*x)
                
        return lst

    # Try greedy first, it's fast
    greedy_sol = greedy_construct()
    
    # Now use SA to improve on greedy_sol or find a better one
    # We can use SA to try to expand the greedy solution or fix it?
    # Actually, SA for fixed size k is better.
    # Let's try to find a solution of size len(greedy_sol) + 1, etc.
    
    # But the loop above already tries sizes 20 down to 15.
    # If greedy found 10, SA might find 12.
    # The loop `for k in range(20, 14, -1)` tries 20, 19, 18, 17, 16, 15.
    # If greedy found 12, and SA failed for 20..15, we return greedy_sol.
    
    return greedy_sol
