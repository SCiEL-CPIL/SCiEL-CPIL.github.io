import time
import random
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    deadline = time.time() + 115  # Safety margin
    
    all_primes = _primes_upto(1000)
    
    # Candidate sets of primes
    # We want to use the full budget ~1.0
    # Strategy: Pick a "backbone" of small primes, then fill with large primes
    
    def fill_budget(backbone, all_primes):
        cost = sum(Fraction(1, p) for p in backbone)
        remaining = Fraction(1, 1) - cost
        if remaining < 0:
            return None
        
        # Add largest primes first (cheapest)
        # Sort primes by value descending
        candidates = [p for p in all_primes if p not in backbone]
        candidates.sort(reverse=True)
        
        current_set = set(backbone)
        for p in candidates:
            if cost + Fraction(1, p) <= Fraction(1, 1):
                current_set.add(p)
                cost += Fraction(1, p)
            else:
                # If we can't add this one, maybe we can add a larger one? 
                # No, we are going down in size, so 1/p is increasing.
                # So if we can't add p, we can't add any smaller prime.
                break
                
        return list(current_set)

    backbones = [
        [2, 3, 7],
        [2, 5, 7, 11],
        [2, 3, 11],
        [3, 5, 7, 11, 13],
        [2, 3, 5], # Will fail budget, but check
        list(p for p in all_primes if p >= 13) # Trimmed version
    ]
    
    # Filter valid backbones and generate full sets
    prime_sets = []
    for b in backbones:
        # Check if backbone itself fits
        cost_b = sum(Fraction(1, p) for p in b)
        if cost_b <= Fraction(1, 1):
            full_set = fill_budget(b, all_primes)
            if full_set:
                prime_sets.append(full_set)
        else:
            # Try to trim backbone? 
            # For now, skip invalid backbones
            pass
            
    # If no sets, use default
    if not prime_sets:
        prime_sets = [list(p for p in all_primes if p >= 13)]
        
    MAX_L = 5000
    best_L = 0
    best_res = None
    
    for p_set in prime_sets:
        N = len(p_set)
        residues = [0] * N
        counts = [0] * (MAX_L + 2)
        
        # Greedy Initialization
        # Sort primes by size? Or random?
        # Let's try to cover 1, 2, 3...
        # For each n, if not covered, find a prime to assign to n.
        
        # To make it faster, we'll just do a few random restarts of SA
        
        for restart in range(10):
            if time.time() > deadline - 10:
                break
                
            # Random init
            for i in range(N):
                residues[i] = random.randint(0, p_set[i] - 1)
                
            # Build counts
            counts = [0] * (MAX_L + 2)
            for i in range(N):
                p = p_set[i]
                r = residues[i]
                n = r if r > 0 else p
                while n <= MAX_L:
                    counts[n] += 1
                    n += p
                    
            # Calculate L
            L = 0
            while L < MAX_L and counts[L+1] > 0:
                L += 1
                
            if L > best_L:
                best_L = L
                best_res = (p_set[:], residues[:])
                
            # Simulated Annealing
            T = 10.0
            cooling = 0.9999
            min_T = 0.1
            current_L = L
            
            # Precompute step costs? No, dynamic.
            
            while T > min_T and time.time() < deadline - 5:
                # Pick random prime
                i = random.randint(0, N-1)
                p = p_set[i]
                old_r = residues[i]
                new_r = random.randint(0, p-1)
                
                if old_r == new_r:
                    continue
                    
                # Calculate new L
                # We need to update counts and find new L.
                # This is expensive if we do it naively.
                
                # Optimization: 
                # Only update counts for n in [1, MAX_L]
                # Decrement old, increment new
                
                # To avoid full recount, we track the first gap.
                # But changing one prime can affect many gaps.
                
                # Let's just do the full update, it's O(MAX_L) per move.
                # MAX_L = 5000. Moves ~ 10000. 50M ops. OK in Python? 
                # 50M ops might take 5-10s.
                # We have 100s. So ~10-20 restarts.
                
                # Apply change
                n = old_r
                if n == 0: n = p
                while n <= MAX_L:
                    counts[n] -= 1
                    n += p
                    
                n = new_r
                if n == 0: n = p
                while n <= MAX_L:
                    counts[n] += 1
                    n += p
                    
                # Find new L
                new_L = 0
                while new_L < MAX_L and counts[new_L+1] > 0:
                    new_L += 1
                    
                delta_L = new_L - current_L
                
                if delta_L > 0:
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_res = (p_set[:], residues[:])
                else:
                    # Accept with probability exp(delta_L / T)
                    # delta_L is negative
                    if random.random() < pow(2.718281828, delta_L / T):
                        current_L = new_L
                        residues[i] = new_r
                    else:
                        # Revert
                        n = new_r
                        if n == 0: n = p
                        while n <= MAX_L:
                            counts[n] -= 1
                            n += p
                            
                        n = old_r
                        if n == 0: n = p
                        while n <= MAX_L:
                            counts[n] += 1
                            n += p
                
                T *= cooling
                
    # Return best
    if best_res is None:
        # Fallback
        return [[2, 0]]
        
    p_set, residues = best_res
    return [[p_set[i], residues[i]] for i in range(len(p_set))]
