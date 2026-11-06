import random
import time

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    TARGET_L = 3000
    WORK_MAX = TARGET_L

    best_L = 0
    best_res = None
    
    start_time = time.time()
    time_budget = 115
    
    attempts = 0
    while time.time() - start_time < time_budget and attempts < 10:
        attempts += 1
        
        if attempts == 1:
            residues = [0] * n_primes
        else:
            residues = [random.randint(0, p - 1) for p in primes]
            
        cov = [0] * (WORK_MAX + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x_val = start
            while x_val <= WORK_MAX:
                cov[x_val] += 1
                x_val += p
                
        stack = []
        L = 0
        x = 1
        success = False
        
        while x <= WORK_MAX:
            if cov[x] >= 2:
                L = x
                x += 1
                continue
            
            candidates = []
            for i in range(n_primes):
                cur_r = residues[i]
                new_r = x % primes[i]
                if new_r == cur_r:
                    continue
                
                cost = 0
                p = primes[i]
                if cur_r == 0:
                    start_y = p
                else:
                    start_y = cur_r
                y = start_y
                while y < x:
                    if cov[y] == 2:
                        cost += 1
                        break
                    y += p
                
                if cost == 0:
                    candidates.append((i, new_r))
            
            if not candidates:
                if not stack:
                    break
                
                last = stack.pop()
                x_decided, idx, old_r, new_r_used, cov_snap, res_snap, untried = last
                
                cov[:] = cov_snap
                residues[:] = res_snap
                x = x_decided
                
                if untried:
                    i_next, r_next = untried[0]
                    untried_remaining = untried[1:]
                    
                    # Apply change
                    p = primes[i_next]
                    # Remove old (which is new_r_used from the popped decision? No, the state is restored to BEFORE the decision)
                    # So residues[idx] is the OLD residue from before the decision.
                    # We are now applying i_next, r_next.
                    # Note: i_next might be the same as idx?
                    # Yes, if we are trying a different residue for the same prime.
                    # Or a different prime.
                    
                    # We need to apply the change relative to the RESTORED state.
                    # The restored state has residues[idx] = old_r (from the entry).
                    # We are changing i_next to r_next.
                    
                    # Helper to apply change given current state
                    def apply_change_state(i, new_r, res, cov):
                        p_val = primes[i]
                        old_r_val = res[i]
                        
                        if old_r_val == 0:
                            start = p_val
                        else:
                            start = old_r_val
                        y_val = start
                        while y_val <= WORK_MAX:
                            cov[y_val] -= 1
                            y_val += p_val
                            
                        if new_r == 0:
                            start = p_val
                        else:
                            start = new_r
                        y_val = start
                        while y_val <= WORK_MAX:
                            cov[y_val] += 1
                            y_val += p_val
                            
                        res[i] = new_r

                    apply_change_state(i_next, r_next, residues, cov)
                    
                    stack.append((x, i_next, r_next, cov[:], residues[:], untried_remaining))
                    continue
                else:
                    continue
            
            # Have candidates
            cov_snap = cov[:]
            res_snap = residues[:]
            
            if len(candidates) > 1:
                candidates.sort(key=lambda t: primes[t[0]], reverse=True)
                i_best, r_best = candidates[0]
                others = candidates[1:]
            else:
                i_best, r_best = candidates[0]
                others = []
            
            # Apply
            def apply_change_state(i, new_r, res, cov):
                p_val = primes[i]
                old_r_val = res[i]
                
                if old_r_val == 0:
                    start = p_val
                else:
                    start = old_r_val
                y_val = start
                while y_val <= WORK_MAX:
                    cov[y_val] -= 1
                    y_val += p_val
                    
                if new_r == 0:
                    start = p_val
                else:
                    start = new_r
                y_val = start
                while y_val <= WORK_MAX:
                    cov[y_val] += 1
                    y_val += p_val
                    
                res[i] = new_r

            apply_change_state(i_best, r_best, residues, cov)
            
            if others:
                stack.append((x, i_best, r_best, cov_snap, res_snap, others))
        
        if L > best_L:
            best_L = L
            best_res = residues[:]
            
    if best_res is None:
        return [0] * 168
        
    return best_res
