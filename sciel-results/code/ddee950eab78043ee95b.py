import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    MAX_VAL = 10000
    N = 12
    
    # Precompute smallest prime factor up to 2*MAX_VAL
    MAX_SUM = 2 * MAX_VAL
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        if n == 1:
            return set()
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def evaluate(a):
        primes = set()
        for i in range(N):
            for j in range(i+1, N):
                s = a[i] + a[j]
                primes.update(get_prime_factors(s))
        return primes

    best_primes = None
    best_set = None
    
    # Strategy 1: Random restarts with local search
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize
        # Try to start with a structure that might have few primes
        # e.g., multiples of a small number
        base = random.randint(1, 100)
        a = []
        used = set()
        while len(a) < N:
            val = random.randint(1, MAX_VAL)
            if val not in used:
                used.add(val)
                a.append(val)
        
        current_primes = evaluate(a)
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove a prime
            # Pick a prime that appears in fewest pairs? Or just iterate
            # To be efficient, we can check if removing a specific element helps
            # Let's try swapping each element with a random candidate
            # Or better: for each prime p in current_primes, try to eliminate it
            
            # Shuffle indices to avoid bias
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Try to find a replacement for a[i]
                # We want to reduce the prime set
                # Let's try a few random candidates
                for _ in range(20):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in used:
                        continue
                    
                    # Check if new_val is distinct
                    # Temporarily swap
                    old_val = a[i]
                    a[i] = new_val
                    used.remove(old_val)
                    used.add(new_val)
                    
                    new_primes = evaluate(a)
                    
                    # Restore
                    a[i] = old_val
                    used.remove(new_val)
                    used.add(old_val)
                    
                    if len(new_primes) < len(current_primes):
                        # Accept improvement
                        a[i] = new_val
                        used.remove(old_val)
                        used.add(new_val)
                        current_primes = new_primes
                        improved = True
                        break # Restart inner loop or continue? Break candidate loop, continue index loop? 
                        # Actually, if we improve, the set of primes changed. 
                        # Let's break out of the candidate loop and re-evaluate indices?
                        # Simpler: just set improved=True and break candidate loop.
                        # The outer while loop will re-shuffle and try again.
                        break
                    
                    # Also check if we can maintain same size but different primes? 
                    # No, we want to minimize size.
                    
            # If no improvement found in this pass, try a random perturbation?
            # Or just break to next restart?
            # If we are stuck, break.
            if not improved:
                break

        # Update best
        if best_primes is None or len(current_primes) < len(best_primes):
            best_primes = current_primes
            best_set = a[:]
            # If we reached 1, we can't do better
            if len(best_primes) == 1:
                return best_set

    # If loop finished, return best
    if best_set:
        return best_set
    
    # Fallback
    return list(range(1, N+1))
