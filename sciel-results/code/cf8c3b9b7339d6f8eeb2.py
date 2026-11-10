import random
import time
import math

def construct():
        N = 192
        MAX_SIZE = 48
        
        # Precompute nothing? The check is fast enough.
        
        def count_violations(S):
            S_list = list(S)
            violations = 0
            for a in S_list:
                R = [x for x in S_list if x != a]
                if len(R) < 3:
                    continue
                
                res_counts = {}
                for x in R:
                    r = x % a
                    res_counts[r] = res_counts.get(r, 0) + 1
                
                bad_found = False
                n_r = len(R)
                for i in range(n_r):
                    b = R[i]
                    rb = b % a
                    for j in range(i+1, n_r):
                        c = R[j]
                        rc = c % a
                        s_mod = (rb + rc) % a
                        target = (-s_mod) % a
                        
                        count_target = res_counts.get(target, 0)
                        if count_target == 0:
                            continue
                        
                        if rb != target and rc != target:
                            bad_found = True
                            break
                        elif rb == target and rc == target:
                            if count_target >= 3:
                                bad_found = True
                                break
                        elif rb == target:
                            if count_target >= 2:
                                bad_found = True
                                break
                        elif rc == target:
                            if count_target >= 2:
                                bad_found = True
                                break
                    if bad_found:
                        break
                if bad_found:
                    violations += 1
            return violations

        def is_valid(S):
            return count_violations(S) == 0

        # Try sizes from MAX_SIZE down to 1
        for target_size in range(MAX_SIZE, 0, -1):
            # Simulated Annealing
            # Initialize with random set of size target_size
            # We can bias towards larger numbers? Or random.
            # Random is fine.
            
            best_set = None
            best_violations = float('inf')
            
            # Multiple restarts
            num_restarts = 5
            for restart in range(num_restarts):
                # Init
                current_set = set(random.sample(range(1, N+1), target_size))
                current_violations = count_violations(current_set)
                
                if current_violations == 0:
                    return list(current_set)
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_set = list(current_set)
                
                # SA Parameters
                T = 1.0
                cooling = 0.99
                min_T = 0.01
                max_steps = 2000 # Adjust based on time
                
                start_time = time.time()
                time_limit = 10.0 # 10s per size attempt? 
                # If we try 48, 47, 46... we need to manage time.
                # Let's allocate time based on remaining budget.
                
                while T > min_T and time.time() - start_time < time_limit:
                    # Generate neighbor
                    # Pick random element in set, replace with random element outside
                    if current_set:
                        elem_in = random.choice(list(current_set))
                        # Find random element not in set
                        # Efficient way: pick random from 1..N, if in set, retry.
                        # Or maintain a list of outside elements.
                        # Since N=192, retry is fine.
                        while True:
                            elem_out = random.randint(1, N)
                            if elem_out not in current_set:
                                break
                        
                        new_set = set(current_set)
                        new_set.remove(elem_in)
                        new_set.add(elem_out)
                        
                        new_violations = count_violations(new_set)
                        delta = new_violations - current_violations
                        
                        if delta <= 0 or random.random() < math.exp(-delta / T):
                            current_set = new_set
                            current_violations = new_violations
                            
                            if current_violations == 0:
                                return list(current_set)
                            
                            if current_violations < best_violations:
                                best_violations = current_violations
                                best_set = list(current_set)
                    
                    T *= cooling
            
            # If we found a valid set, return it
            if best_violations == 0:
                return best_set
        
        # If no valid set found (should not happen), return best effort
        if best_set:
            return best_set
        return [1] # Fallback
