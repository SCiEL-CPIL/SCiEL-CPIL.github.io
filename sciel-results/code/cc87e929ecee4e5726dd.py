import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    N = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # Precompute smallest prime factor (SPF) for fast factorization
    def get_spf(n):
        spf = list(range(n + 1))
        for i in range(2, int(n**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, n+1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    try:
        spf = get_spf(MAX_SUM)
        
        def get_prime_factors(n):
            factors = set()
            while n > 1:
                p = spf[n]
                factors.add(p)
                while n % p == 0:
                    n //= p
            return factors

        def evaluate_set(nums):
            all_primes = set()
            for i in range(N):
                for j in range(i+1, N):
                    s = nums[i] + nums[j]
                    all_primes.update(get_prime_factors(s))
            return len(all_primes)

        # Multi-start local search
        best_set = list(range(1, N + 1))
        best_score = evaluate_set(best_set)
        
        # Try a few random starts
        num_starts = 10
        for k in range(num_starts):
            if time.time() > deadline:
                break
            
            # Generate a random initial set
            # Ensure distinct positive integers <= MAX_VAL
            current_set = random.sample(range(1, MAX_VAL + 1), N)
            current_score = evaluate_set(current_set)
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
            
            # Local search: swap elements
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Shuffle indices to avoid deterministic bias
                indices = list(range(N))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, N):
                        if time.time() > deadline:
                            break
                        
                        # Swap nums[i] and nums[j] with a random new value?
                        # Better: try to replace one element with a random other value
                        # Or swap two existing elements? Swapping doesn't change the set of sums, just indices.
                        # We need to change the values.
                        # Strategy: Pick index i, pick random new_val not in set.
                        # This is O(N^2) per iteration if we check "in set".
                        # Let's stick to: Pick i, pick new_val.
                        
                        pass # We will use a simpler replacement strategy below
            
            # Simpler local search: Random replacement
            for _ in range(200): # Max 200 replacements per start
                if time.time() > deadline:
                    break
                
                idx = random.randint(0, N - 1)
                new_val = random.randint(1, MAX_VAL)
                
                if new_val in current_set:
                    continue
                
                old_val = current_set[idx]
                current_set[idx] = new_val
                
                new_score = evaluate_set(current_set)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_set = current_set[:]
                else:
                    # Revert
                    current_set[idx] = old_val

        return best_set

    except Exception as e:
        # Fallback to a simple valid sequence if anything goes wrong
        return list(range(1, N + 1))

# EVOLVE-BLOCK-END
