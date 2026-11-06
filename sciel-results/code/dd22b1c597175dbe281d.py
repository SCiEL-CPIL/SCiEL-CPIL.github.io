import time
import math
import random
import numpy as np
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    if n >= 0: sieve[0] = False
    if n >= 1: sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    primes = _primes_upto(1000)
    
    # Define candidate sets of primes where sum(1/p) <= 1
    # We want sets that are "dense" enough to cover a long interval.
    # Set A: {3, 5} + primes >= 13. Sum ~ 0.97.
    # Set B: {2} + primes >= 67. Sum ~ 0.95.
    # Set C: {3, 5, 7} + primes >= 150. Sum ~ 0.99.
    
    candidate_sets = [
        [3, 5] + [p for p in primes if p >= 13],
        [2] + [p for p in primes if p >= 67],
        [3, 5, 7] + [p for p in primes if p >= 150],
        [3] + [p for p in primes if p >= 19],
        [5] + [p for p in primes if p >= 11]
    ]
    
    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)
            
    if not valid_sets:
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []
    
    # Allocate time per set
    # We have ~145s. Let's give each set ~25s.
    time_per_set = 25.0
    
    # L_MAX for search. We start with a reasonable size and expand if needed.
    # However, expanding cover_count is expensive. 
    # Let's fix L_MAX to a high value like 20000 to allow for high L, 
    # but be careful with memory and speed.
    # 20000 integers is fine.
    L_MAX = 20000
    
    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        P_sorted = sorted(P)
        
        # 1. Greedy Initialization
        # We want to cover 1..L_MAX.
        # Greedy: for n=1,2,... if not covered, pick a prime and set r = n % p.
        # To improve: pick the prime that maximizes future coverage? 
        # For now, simple greedy: pick the smallest prime not yet assigned?
        # Or pick the prime that has the largest "slack" in coverage?
        # Let's try a heuristic: pick the prime that minimizes the number of 
        # "wasted" slots in the range [1, L_MAX].
        # Actually, just picking the smallest prime is a decent baseline.
        
        residues = {p: 0 for p in P}
        cover_count = np.zeros(L_MAX + 1, dtype=np.int32)
        
        def update_cover(p, r, delta):
            first = r if r > 0 else p
            if first > L_MAX:
                return
            for n in range(first, L_MAX + 1, p):
                cover_count[n] += delta

        # Greedy assignment
        assigned_primes = set()
        # Sort primes by size? Smaller primes cover more densely.
        for n in range(1, L_MAX + 1):
            if cover_count[n] == 0:
                # Find a prime to assign
                # Heuristic: pick the smallest prime not in assigned_primes?
                # This might lead to using small primes for early numbers, 
                # which is good for density.
                chosen_p = None
                for p in P_sorted:
                    if p not in assigned_primes:
                        chosen_p = p
                        break
                
                if chosen_p is None:
                    break # No more primes to assign
                
                r = n % chosen_p
                residues[chosen_p] = r
                assigned_primes.add(chosen_p)
                update_cover(chosen_p, r, 1)
            
            if time.time() > set_deadline:
                break
        
        # Calculate current L
        def get_L():
            # Find first zero in cover_count[1:]
            zeros = np.nonzero(cover_count[1:])[0]
            if len(zeros) == 0:
                return L_MAX
            return zeros[0]
        
        current_L = get_L()
        best_set_L = current_L
        best_set_residues = residues.copy()
        
        # 2. Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        steps = 0
        
        # Precompute indices for faster random access?
        # List of primes for random choice
        P_list = list(P)
        
        while time.time() < set_deadline:
            # Pick a random prime to change
            idx = random.randint(0, len(P_list) - 1)
            p = P_list[idx]
            r_old = residues[p]
            
            # Generate new residue
            # Random walk vs Random Jump?
            # Jump is better for SA to escape local minima.
            r_new = random.randint(0, p - 1)
            
            if r_new == r_old:
                continue
            
            # Update cover_count
            update_cover(p, r_old, -1)
            update_cover(p, r_new, 1)
            
            # Calculate new L
            new_L = get_L()
            
            # Acceptance
            if new_L > current_L:
                current_L = new_L
                residues[p] = r_new
                if current_L > best_set_L:
                    best_set_L = current_L
                    best_set_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta == 0:
                    # Accept equal moves with some probability to explore plateaus
                    if random.random() < 0.3:
                        residues[p] = r_new
                else:
                    # delta < 0
                    # Use a smoother acceptance probability?
                    # Standard SA: exp(delta / T)
                    # If delta is large negative, prob is very small.
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        residues[p] = r_new
                        current_L = new_L
                    else:
                        # Reject: revert cover_count
                        update_cover(p, r_new, -1)
                        update_cover(p, r_old, 1)
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            steps += 1
            
            # Early exit if we've found a very good solution?
            # L >= 999 is the target.
            if best_set_L > 999:
                # We can continue to try to improve, but if we hit a very high L, 
                # we might want to stop early to save time for other sets?
                # No, let's let it run its course for this set.
                pass

        # Update global best
        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[p, best_set_residues[p]] for p in best_set_residues]
            
            # If we found a solution that covers 999, we are doing well.
            # We can break early if we have a very high L, e.g., > 5000?
            # But maybe another set could do even better?
            # Let's not break early unless we are running out of time.

    # Final check and return
    if not best_result_global:
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result_global
