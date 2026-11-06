import time
import random
import math
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    
    # Candidate prime sets where sum(1/p) <= 1
    # We want a set of primes that can potentially cover a long prefix.
    # Small primes are powerful for covering.
    # Set A: {3, 5} + primes >= 13. Sum ~ 0.97.
    # Set B: {2} + primes >= 67. Sum ~ 0.95.
    # Set C: {3, 5, 7} + primes >= 150. Sum ~ 0.99.
    
    candidate_sets = [
        [3, 5] + [p for p in primes if p >= 13],
        [2] + [p for p in primes if p >= 67],
        [3, 5, 7] + [p for p in primes if p >= 150]
    ]
    
    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)
            
    if not valid_sets:
        # Fallback to seed-like if no valid sets found
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []
    
    # Time allocation per set
    time_per_set = 45.0 # 3 sets * 45s = 135s. 10s margin.
    
    L_MAX = 20000 # Cap for coverage calculation. 10^5 is the hard cap, but 20000 is a good target for search.
    
    for P in valid_sets:
        if time.time() > deadline - 5:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        P_sorted = sorted(P)
        
        # 1. Greedy Initialization
        # We want to cover 1..L.
        # Greedy: for n=1,2,... if not covered, pick a prime p and set r = n % p.
        # We track coverage counts to allow O(1) L calculation and fast updates.
        
        residues = {p: 0 for p in P}
        cover_count = np.zeros(L_MAX + 1, dtype=np.int8)
        
        def update_cover(p, r, delta):
            # Update cover_count for prime p with residue r by delta
            # r is in [0, p-1]. The numbers covered are r, r+p, r+2p, ...
            # Note: if r == 0, the first positive number is p.
            # The range is 1..L_MAX.
            first = r if r > 0 else p
            if first > L_MAX:
                return
            # Vectorized update for speed
            indices = np.arange(first, L_MAX + 1, p)
            cover_count[indices] += delta

        # Initialize cover_count to 0
        # Greedy assignment
        # Strategy: For each uncovered n, pick a prime.
        # Heuristic: Pick the prime that minimizes the "cost" or maximizes future potential?
        # Simple heuristic: Pick the smallest prime that hasn't been used recently?
        # Or just cycle through primes?
        # Let's try: Pick the smallest prime p such that n % p is "good".
        # Actually, since we just need to cover n, any p works.
        # To maximize coverage length, we should prefer smaller primes as they have higher density.
        # So, pick the smallest available prime? But we can reuse primes!
        # Wait, the problem says "distinct primes p".
        # "construct() -> classes [[p, r], ...] with distinct primes p <= 1000"
        # So each prime can appear at most once.
        
        # This changes the greedy strategy significantly.
        # We have a set of primes P. We assign each prime p a residue r_p.
        # The union of classes {n : n % p == r_p} must cover 1..L.
        
        # Greedy:
        # Start with L=0.
        # While there is an uncovered number n <= L_MAX:
        #   Find the smallest uncovered n.
        #   Pick a prime p from P that is not yet assigned.
        #   Set r_p = n % p.
        #   Update coverage.
        
        # Which prime to pick?
        # We want to cover as much as possible.
        # Smaller primes cover more numbers in the range 1..L_MAX.
        # So we should prioritize assigning smaller primes.
        
        # Let's sort P in ascending order.
        # Iterate n from 1 to L_MAX.
        # If n is not covered:
        #   Pick the smallest unassigned prime p.
        #   Set r_p = n % p.
        #   Mark p as assigned.
        #   Update coverage.
        
        assigned_primes = set()
        
        # Optimization: Precompute n % p for all p? No, just do it on the fly.
        
        for n in range(1, L_MAX + 1):
            if cover_count[n] == 0:
                # Find smallest unassigned prime
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
            # Find the first index where cover_count is 0
            # cover_count is 1-indexed effectively (index 0 is unused)
            # We care about 1..L_MAX
            zeros = np.where(cover_count[1:L_MAX+1] == 0)[0]
            if len(zeros) == 0:
                return L_MAX
            return zeros[0]
        
        current_L = get_L()
        best_set_L = current_L
        best_set_residues = residues.copy()
        
        # 2. Local Search / Simulated Annealing
        # We have assigned all primes we could.
        # We can change residues of assigned primes.
        # We can also potentially swap primes? No, the set P is fixed for this run.
        # But we can change which prime is assigned to which "role"?
        # Actually, the set of primes is fixed. The residues are the variables.
        
        # Heuristic: Target the first gap.
        # If L is current, gap is L+1.
        # We want to cover L+1.
        # We can change the residue of any assigned prime p to (L+1) % p.
        # This will cover L+1. But it might uncover other numbers.
        # So we need to check the new L.
        
        # Let's do a hill-climbing with random restarts/perturbations.
        
        T = 5.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < set_deadline:
            # Pick a random assigned prime
            assigned_list = list(assigned_primes)
            if not assigned_list:
                break
            p = random.choice(assigned_list)
            r_old = residues[p]
            
            # Generate new residue
            # Strategy 1: Random
            # Strategy 2: Target the first gap
            # Let's mix them.
            if random.random() < 0.5:
                # Target first gap
                target = current_L + 1
                if target <= L_MAX:
                    r_new = target % p
                else:
                    r_new = random.randint(0, p - 1)
            else:
                r_new = random.randint(0, p - 1)
                
            if r_new == r_old:
                continue
            
            # Update cover_count
            update_cover(p, r_old, -1)
            update_cover(p, r_new, 1)
            
            new_L = get_L()
            
            # Acceptance
            if new_L > current_L:
                current_L = new_L
                residues[p] = r_new
                if current_L > best_set_L:
                    best_set_L = current_L
                    best_set_residues = residues.copy()
            else:
                # Accept with probability exp((new_L - current_L) / T)
                delta = new_L - current_L
                if delta == 0:
                    if random.random() < 0.5:
                        residues[p] = r_new
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        residues[p] = r_new
                    else:
                        # Reject: revert cover_count
                        update_cover(p, r_new, -1)
                        update_cover(p, r_old, 1)
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Update global best
        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[p, best_set_residues[p]] for p in best_set_residues if p in assigned_primes]
            
            # Check if we should break early
            if best_L_global >= 999:
                # We found a good solution, but let's see if we can push higher.
                # If we have time, continue.
                pass

    # Final check and return
    if not best_result_global:
        # Fallback to seed
        primes_seed = [p for p in primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result_global
