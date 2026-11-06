import time
import random
import math
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin for 120s limit
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    all_primes = get_primes(1000)
    
    # Build prime set: include 11, remove largest primes until sum(1/p) <= 1
    candidates = [p for p in all_primes if p >= 11]
    candidates_sorted_desc = sorted(candidates, reverse=True)
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    
    while current_sum > 1:
        if not candidates_sorted_desc:
            break
        p_to_remove = candidates_sorted_desc.pop()
        primes_set.remove(p_to_remove)
        current_sum -= Fraction(1, p_to_remove)
        
    primes_set = sorted(primes_set)
    n_primes = len(primes_set)
    
    L_CHECK_LIMIT = 5000
    L_FINAL_LIMIT = 100000
    
    # Precompute for fast coverage evaluation
    # For each prime index, precompute the range of indices in 1..L_CHECK_LIMIT it covers
    # We'll use a numpy array for cover counts
    
    def evaluate_coverage(residues):
        """Fast coverage evaluation using numpy. Returns L (first uncovered)."""
        cover = np.zeros(L_CHECK_LIMIT + 1, dtype=np.int8)
        for i, p in enumerate(primes_set):
            r = residues[i]
            start = r if r != 0 else p
            if start <= L_CHECK_LIMIT:
                # Mark all positions start, start+p, start+2p, ...
                cover[start::p] += 1
        
        # Find first zero
        zeros = np.where(cover[1:] == 0)[0]
        if len(zeros) == 0:
            return L_CHECK_LIMIT
        return int(zeros[0])
    
    def init_residues_greedy():
        """Greedy initialization: assign residues to cover 1, 2, 3, ..."""
        res = [0] * n_primes
        for i, p in enumerate(primes_set):
            target = i + 1
            res[i] = target % p
        return res
    
    best_L = 0
    best_classes = None
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9999
    
    residues = init_residues_greedy()
    current_L = evaluate_coverage(residues)
    
    if current_L > best_L:
        best_L = current_L
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
    
    # Precompute random residue options for each prime
    # To speed up, we'll just use random.randint in the loop
    
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            T = T_min
        
        # Pick a random prime index to change
        idx = random.randint(0, n_primes - 1)
        p = primes_set[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Apply change
        residues[idx] = new_r
        new_L = evaluate_coverage(residues)
        
        delta = new_L - current_L
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
        else:
            # Reject: revert
            residues[idx] = old_r
        
        T *= cooling
    
    # Final polish: greedy hill-climbing from best solution
    residues = [r for _, r in best_classes]
    current_L = evaluate_coverage(residues)
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for idx in range(n_primes):
            p = primes_set[idx]
            old_r = residues[idx]
            # Try all residues for this prime
            for new_r in range(p):
                if new_r == old_r:
                    continue
                residues[idx] = new_r
                new_L = evaluate_coverage(residues)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                    break
            residues[idx] = old_r
    
    # Final Evaluation
    if best_classes is None:
        best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
    
    # Calculate exact L up to L_FINAL_LIMIT
    cover_count_final = [0] * (L_FINAL_LIMIT + 1)
    for p, r in best_classes:
        start = r
        if start == 0:
            start = p
        if start <= L_FINAL_LIMIT:
            for n in range(start, L_FINAL_LIMIT + 1, p):
                cover_count_final[n] += 1
                
    final_L = 0
    for n in range(1, L_FINAL_LIMIT + 1):
        if cover_count_final[n] == 0:
            break
        final_L = n
        
    return best_classes

# EVOLVE-BLOCK-END
