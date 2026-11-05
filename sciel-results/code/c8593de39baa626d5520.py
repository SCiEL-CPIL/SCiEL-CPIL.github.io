import time
import random
import math

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(math.isqrt(p)) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    # L_max for evaluation
    L_max = 50000 
    # Cap at 10^5 as per problem statement, but 50k is usually enough for SA to find good L
    # If we find L > 50000, we can expand, but let's start here.
    
    # Initialize residues
    r = [0] * n_primes
    
    # Coverage array
    cov = [0] * (L_max + 1)
    
    # Set of indices where cov[x] < 2
    bad_set = set(range(1, L_max + 1)) # Initially all are bad (cov=0)
    
    # Helper to update coverage and bad_set
    def update(p, r_old, r_new, sign):
        # sign: +1 for adding coverage, -1 for removing
        # We need to remove old residue contribution and add new?
        # Actually, let's just apply the change.
        pass

    # Better: direct application of residue change
    def apply_change(idx, new_r):
        p = primes[idx]
        old_r = r[idx]
        if old_r == new_r:
            return
        
        # Remove old coverage
        x = old_r
        while x <= L_max:
            if x > 0: # x starts from old_r, if old_r==0, x=0 is not in 1..L_max
                cov[x] -= 1
                if cov[x] < 2:
                    bad_set.add(x)
                elif cov[x] == 2:
                    bad_set.discard(x)
            x += p
            
        # Add new coverage
        r[idx] = new_r
        x = new_r
        while x <= L_max:
            if x > 0:
                cov[x] += 1
                if cov[x] == 2:
                    bad_set.discard(x)
                elif cov[x] < 2:
                    bad_set.add(x)
            x += p

    def get_L():
        if not bad_set:
            return L_max
        return min(bad_set) - 1

    # Initial coverage with r=0
    for i, p in enumerate(primes):
        x = 0
        while x <= L_max:
            if x > 0:
                cov[x] += 1
                if cov[x] == 2:
                    bad_set.discard(x)
            x += p

    current_L = get_L()
    
    # Greedy Phase
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    while time.time() < deadline and current_L < L_max:
        x_fail = min(bad_set)
        
        # Find a prime p that does not cover x_fail
        # And check if changing it fixes x_fail without breaking previous L
        candidates = []
        for i, p in enumerate(primes):
            if x_fail % p != r[i]:
                # Check if changing r[i] to x_fail % p is safe
                # It affects multiples of p.
                # We need to ensure no x in [1, current_L] drops below 2
                # Since x_fail is the FIRST bad one, all x < x_fail are good (cov >= 2).
                # Changing r[i] will change cov for x = k*p.
                # If x < x_fail, cov[x] was >= 2.
                # If we remove old coverage, cov[x] might become 1.
                # So we must check if old_r[i] covers any x < x_fail.
                # If old_r[i] covers x, and cov[x] == 2, it will break.
                # So we need to check all x < x_fail that are multiples of p and congruent to old_r[i].
                # If any such x has cov[x] == 2, it's a break.
                
                safe = True
                x = old_r_i = r[i]
                while x < x_fail:
                    if cov[x] == 2:
                        safe = False
                        break
                    x += p
                
                if safe:
                    # Also, does it fix x_fail?
                    # We set new_r = x_fail % p.
                    # x_fail will get +1 coverage.
                    # Since x_fail was bad (cov < 2), it becomes good or better.
                    candidates.append(i)
        
        if candidates:
            # Pick a candidate, maybe the one with largest p (least disruption globally)
            # Or random
            i = random.choice(candidates)
            new_r = x_fail % primes[i]
            apply_change(i, new_r)
            current_L = get_L()
        else:
            # Stuck in Greedy. Break to SA?
            # If we break, we might miss improvements.
            # Let's just break Greedy and go to SA.
            break

    # SA Phase
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick random index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = r[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        # Calculate Delta L
        # We need to simulate the change
        # To avoid modifying state and reverting, we can just do it and revert if rejected.
        
        # Apply
        apply_change(i, new_r)
        new_L = get_L()
        
        delta_L = new_L - current_L
        
        if delta_L > 0 or random.random() < math.exp(delta_L / T):
            current_L = new_L
        else:
            # Revert
            apply_change(i, old_r)
            
        T *= cooling
        
    return r
