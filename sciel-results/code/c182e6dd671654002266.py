import math
import random
import time

def construct():
        N = 256
        # Sieve of Eratosthenes
        sieve = [True] * (N + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(N**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, N + 1, i):
                    sieve[j] = False
        primes = [i for i in range(2, N + 1) if sieve[i]]
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
            
        # Precompute prime factors for each number
        factors = [[] for _ in range(N + 1)]
        for p in primes:
            for multiple in range(p, N + 1, p):
                factors[multiple].append(p)
                
        # Initial set: all primes + 1
        current_set = set(primes + [1])
        current_sum = sum(current_set)
        best_set = set(current_set)
        best_sum = current_sum
        
        def get_conflicts(num):
            # Returns list of numbers in current_set that share a factor with num
            # Exclude 1
            confs = []
            for s in current_set:
                if s == 1: continue
                if gcd(num, s) > 1:
                    confs.append(s)
            return confs
            
        start_time = time.time()
        max_time = 80.0 # Safety margin
        
        while time.time() - start_time < max_time:
            improved = False
            # Find best swap
            best_gain = 0
            best_candidate = None
            best_conflicts = None
            
            # Check all composites
            for c in range(4, N + 1):
                if c in current_set:
                    continue
                confs = get_conflicts(c)
                # If c is coprime to all, confs is empty.
                # Gain is c - sum(confs)
                gain = c - sum(confs)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                    best_conflicts = confs
                    
            if best_candidate is not None:
                # Perform swap
                for s in best_conflicts:
                    current_set.remove(s)
                current_set.add(best_candidate)
                current_sum += best_gain
                improved = True
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = set(current_set)
            else:
                # Local optimum reached. Perturb.
                # Randomly remove a prime and re-add a random multiple?
                # Or just break and try a different random restart?
                # Let's do a random kick:
                # Pick a random prime in the set, remove it, and add a random multiple of it (if valid).
                # Or simpler: Randomly swap two elements?
                
                # Let's try a random restart if we are stuck for a while?
                # Or just break and rely on multiple runs?
                # Since we can't run multiple times in one construct call easily without overhead,
                # let's do a random perturbation.
                
                # Pick a random number from the set (excluding 1)
                if len(current_set) > 1:
                    to_remove = random.choice([x for x in current_set if x != 1])
                    current_set.remove(to_remove)
                    current_sum -= to_remove
                    
                    # Try to add a random number coprime to the rest
                    # Just add the largest available number coprime to the rest?
                    # Or random.
                    # Let's add a random prime not in the set?
                    # No, that might not improve sum.
                    
                    # Let's just break and return best so far.
                    # The greedy usually finds a good solution quickly.
                    break

        return list(best_set)
