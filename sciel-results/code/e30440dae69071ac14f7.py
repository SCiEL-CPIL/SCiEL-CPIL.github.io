"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate primes up to 1000
    primes = _sieve(1000)
    
    # 2. Initialize with a valid, decent heuristic (similar to seed but maybe better ordered)
    # The seed uses primes >= 13. Let's start with all primes >= 5 or so, 
    # but we need sum(1/p) <= 1.
    # Sum of reciprocals of primes up to 1000 is approx 2.2.
    # We need to select a subset.
    
    # Heuristic initialization:
    # Start with primes in descending order? Or ascending?
    # Ascending primes cover small numbers densely but use budget fast?
    # Actually, 1/p is larger for small p. So small p are "expensive".
    # To maximize L, we want to cover 1..L.
    # Small residues on small primes are powerful.
    
    # Let's try a greedy initialization that picks primes to cover the next uncovered integer.
    # But first, let's just pick a random valid subset and random residues, then optimize.
    
    # A known good set of primes for covering systems often involves small primes.
    # Let's include primes up to 47 or so, and some larger ones.
    # Sum of 1/p for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] is approx 1.0-1.2.
    # We must be careful with the budget.
    
    # Strategy:
    # 1. Select a subset of primes P such that sum(1/p) <= 1.
    # 2. Assign residues r_p in [0, p-1].
    # 3. Optimize r_p to maximize L.
    
    # Let's fix the set of primes first to a "standard" covering set or optimize it too?
    # Optimizing the set is harder. Let's stick to a set of primes that has a good chance.
    # The seed used primes starting from 13.
    # Let's use primes from 5 to 1000, but prune to fit budget.
    
    # Greedy selection of primes to maximize potential coverage:
    # Start with empty set, sum=0.
    # Add primes in increasing order? No, small p have high cost.
    # Maybe we should just use a fixed set of small primes that fits the budget, 
    # and then use local search on residues.
    
    # Let's try to include primes 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47.
    # Sum 1/p for these:
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47
    # Let's calculate exact sum.
    
    candidate_primes = [p for p in primes if p <= 100]
    # Sort by 1/p descending (i.e., p ascending) to add "powerful" ones first?
    # Or just take the smallest ones until budget is tight?
    
    selected_primes = []
    current_sum = Fraction(0, 1)
    # We want to cover 1..L. Small primes are essential for density.
    # Let's iterate primes in ascending order and add if budget allows.
    for p in primes:
        cost = Fraction(1, p)
        if current_sum + cost <= Fraction(1, 1):
            # Heuristic: Don't take ALL small primes if it blocks larger ones?
            # Actually, for covering 1..L, small primes are very efficient at covering "gaps"
            # created by larger primes? No, small primes cover more numbers per unit length.
            # Let's just take the smallest primes that fit.
            selected_primes.append(p)
            current_sum += cost
    
    # If we only took small primes, L might be small. 
    # The seed took primes >= 13. Sum was ~0.92.
    # Let's check if we can add more primes.
    # The above loop takes 2, 3, 5, 7, 11, 13... until sum > 1.
    # Sum 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47...
    # 1/2=0.5
    # +1/3=0.833
    # +1/5=1.033 -> Exceeds 1.
    # So this greedy would only select {2, 3}. That's bad.
    
    # Better initialization:
    # Skip 2 and 3?
    # If we skip 2 and 3, we can take 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53...
    # Sum 1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53
    # 0.2+0.142+0.09+0.076+0.058+0.052+0.043+0.034+0.032+0.027+0.024+0.023+0.021+0.018
    # Sum approx 0.81. We have room for more.
    
    # Let's try a set: Primes from 5 to 1000, selected greedily by smallest p first, 
    # but skipping 2 and 3 to save budget for more primes?
    # Or maybe include 2 but skip 3?
    # Let's try to maximize the number of primes while keeping sum <= 1.
    # This is a knapsack-like problem.
    
    # For now, let's use a fixed set that is known to be reasonable:
    # Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Let's check sum.
    # Instead of hardcoding, let's run a quick greedy selection in the code.
    
    def select_primes_greedy():
        # Sort primes by 1/p descending (smallest p first)
        # But we know 2 and 3 are too expensive if we want many primes.
        # Let's try subsets.
        # Option A: Include 2, 3. Remaining budget 1/6.
        #   Next primes: 5 (1/5=0.2), 7(0.14), 11(0.09), 13(0.076), 17(0.058), 19(0.052)
        #   1/5+1/7+1/11+1/13+1/17+1/19 = 0.2+0.142+0.09+0.076+0.058+0.052 = 0.618.
        #   1/6 = 0.166. So we can't even take 5 if we take 2 and 3.
        #   So {2,3} is very restrictive.
        
        # Option B: Include 2, skip 3. Remaining budget 1/2.
        #   Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47...
        #   Sum 1/5..1/47 is approx 0.8. 1/2 is 0.5. So we can take a few.
        
        # Option C: Skip 2, 3. Remaining budget 1.
        #   Primes: 5, 7, 11, ...
        #   Sum 1/5..1/100 is approx 0.8. Sum 1/5..1/200 is approx 0.9.
        #   This allows many primes.
        
        # Let's go with Option C: Primes >= 5.
        # Greedy: take smallest primes first until sum > 1.
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < 5:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
            else:
                # Try to fill remaining budget with larger primes?
                # Or just stop.
                # Let's stop for now.
                pass
        return sel, s

    # Actually, the seed used primes >= 13.
    # Let's use primes >= 5 for more flexibility.
    selected_primes, current_sum = select_primes_greedy()
    
    # If selected_primes is empty or small, fallback to seed logic
    if len(selected_primes) < 10:
         selected_primes = [p for p in primes if p >= 13]
         # Re-check sum
         current_sum = sum(Fraction(1, p) for p in selected_primes)
         while current_sum > Fraction(1, 1) and selected_primes:
             # Remove largest prime (smallest cost) to save budget? No, remove smallest prime (largest cost)?
             # To maximize count, remove smallest prime? No, to fit budget, remove any.
             # Let's just remove the last one added.
             p = selected_primes.pop()
             current_sum -= Fraction(1, p)
             
    # Initialize residues randomly
    residues = {p: random.randint(0, p-1) for p in selected_primes}
    
    # Function to calculate L
    def calc_L(res_map):
        # Create a boolean array for 1..100000
        # But we only need to check up to some limit.
        # The score is capped at 10^5.
        # We can stop early if we find a gap.
        covered = [False] * 100001
        for p, r in res_map.items():
            for n in range(r % p, 100001, p):
                if n > 0: # We care about 1..L
                    covered[n] = True
        
        L = 0
        for i in range(1, 100001):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    # This calc_L is O(N * K) which might be slow if called many times.
    # N=10^5, K~20-50. 5*10^6 ops per call.
    # 110 seconds / (5*10^6) ~ 20 calls. Too slow for SA.
    
    # Optimization:
    # Instead of updating the full array, we can maintain a "gap" structure or just check locally.
    # Or, we can use a bitset (numpy) for speed.
    
    import numpy as np
    N = 100000
    covered_bits = np.zeros(N+1, dtype=bool)
    
    def update_coverage(res_map):
        nonlocal covered_bits
        covered_bits[:] = False
        for p, r in res_map.items():
            start = r % p
            if start == 0:
                start = p
            # np.arange(start, N+1, p)
            covered_bits[start:N+1:p] = True
        # Find first False in 1..N
        # np.argmax(covered_bits[1:] == False)
        # If all True, argmax returns 0? No, if all True, argmax(1-arr) is 0?
        # Let's use:
        # idx = np.where(covered_bits[1:] == False)[0]
        # if len(idx) > 0: return idx[0]
        # else: return N
        
        # Faster:
        # gaps = np.flatnonzero(~covered_bits[1:])
        # if len(gaps) > 0:
        #     return gaps[0]
        # else:
        #     return N
        
        # Even faster:
        # Find first False in covered_bits[1:]
        # We can just loop in Python if N is small enough? 10^5 is small.
        # But numpy is better.
        
        # Let's stick to a simple loop for L calculation but only up to current best L + margin.
        # No, we need to know if we extended.
        
        pass

    # Let's implement a faster L calculator using numpy
    def get_L(res_map):
        nonlocal covered_bits
        covered_bits[:] = False
        for p, r in res_map.items():
            start = r % p
            if start == 0:
                start = p
            if start <= N:
                covered_bits[start:N+1:p] = True
        
        # Find first gap
        # We check 1..N
        # If covered_bits[i] is False for any i in 1..N, L is i-1.
        # We can find the first False in covered_bits[1:]
        # np.argmax returns index of first True.
        # We want first False.
        # ~covered_bits[1:] gives True for gaps.
        # If no gap, all False in ~covered_bits.
        
        # To avoid creating new arrays:
        # We can just iterate?
        # 10^5 iterations is ~1ms in Python.
        # 110s / 1ms = 110,000 iterations. This is feasible!
        
        L = 0
        for i in range(1, N+1):
            if covered_bits[i]:
                L = i
            else:
                break
        return L

    # Simulated Annealing
    current_L = get_L(residues)
    best_L = current_L
    best_residues = residues.copy()
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random prime to change
        p = random.choice(selected_primes)
        old_r = residues[p]
        new_r = random.randint(0, p-1)
        if new_r == old_r:
            continue
            
        residues[p] = new_r
        
        new_L = get_L(residues)
        
        delta = new_L - current_L
        
        if delta > 0:
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            if random.random() < np.exp(delta / T):
                current_L = new_L
            else:
                residues[p] = old_r # Revert
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return best found
    return [[p, best_residues[p]] for p in selected_primes]
# EVOLVE-BLOCK-END
