import time
import random
import math

TIME_LIMIT = 85.0

def _init_sieve(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Precompute smallest prime factor up to 20000
# Max sum of two numbers <= 10000 is 20000.
# In our scaled approach, if x_i = B * u_i, and x_i <= 10000, then u_i <= 10000/B.
# Sum u_i+u_j <= 20000/B.
# Sum x_i+x_j <= 20000.
# So we need prime factors of numbers up to 20000.
LIMIT = 20000
spf = _init_sieve(LIMIT)

def get_prime_factors(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def score_numbers(nums):
    """Count distinct prime factors of all pairwise sums."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            primes |= get_prime_factors(s)
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = float('inf')
    best_nums = list(range(1, 13)) # Fallback
    
    # Candidate bases B such that x_i = B * u_i
    # We want B to have few prime factors.
    candidates_B = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 120, 240, 480, 960, 1920, 4800]
    
    # Shuffle to avoid bias, or just iterate
    random.seed(42)
    
    # We will run local search for each B
    # For a given B, u_i must be distinct, 1 <= u_i <= 10000 // B
    # Sums x_i + x_j = B * (u_i + u_j)
    # Prime factors of x_i + x_j are primes(B) U primes(u_i + u_j)
    
    for B in candidates_B:
        if time.time() > deadline - 5:
            break
            
        max_u = 10000 // B
        if max_u < 12:
            continue
            
        primes_B = get_prime_factors(B)
        
        # Initialize u's
        # Try to start with small numbers
        # A good start might be multiples of some small number to keep sums smooth?
        # Or just random.
        
        # Let's try a few random restarts for each B
        for restart in range(5):
            if time.time() > deadline - 2:
                break
                
            # Initialize u
            # Strategy: pick 12 distinct integers from [1, max_u]
            # To help convergence, maybe pick from a smaller range first?
            # If max_u is large, random picking from [1, max_u] might be slow to converge to smooth sums.
            # Let's pick from [1, min(max_u, 100)] initially?
            # But we need distinct.
            
            if max_u <= 100:
                pool = list(range(1, max_u + 1))
                random.shuffle(pool)
                u = pool[:12]
            else:
                # Pick 12 random distinct
                u = random.sample(range(1, max_u + 1), 12)
            
            current_u = u[:]
            current_nums = [val * B for val in current_u]
            current_score = score_numbers(current_nums)
            
            no_improve = 0
            max_iter = 2000
            
            while no_improve < 100 and max_iter > 0 and time.time() < deadline - 2:
                max_iter -= 1
                idx = random.randint(0, 11)
                old_val = current_u[idx]
                
                # Try to replace with a new value
                # Bias towards smaller values? Smaller sums -> smaller primes.
                # But we need distinct.
                # Let's pick a random new value from [1, max_u]
                # Or pick from a set of "smooth" candidates?
                
                new_val = random.randint(1, max_u)
                if new_val in current_u:
                    no_improve += 1
                    continue
                
                current_u[idx] = new_val
                test_nums = [val * B for val in current_u]
                test_score = score_numbers(test_nums)
                
                if test_score < current_score:
                    current_score = test_score
                    no_improve = 0
                else:
                    current_u[idx] = old_val
                    no_improve += 1
            
            if current_score < best_score:
                best_score = current_score
                best_nums = [val * B for val in current_u]
                
                # If we found a very good score, we might stop early?
                # Minimum possible is 1 (all sums power of 2? unlikely for 12 distinct).
                # 2 is possible (primes 2,3).
                # 3 is possible (primes 2,3,5).
                if best_score <= 3:
                    # Very good, break out of loops?
                    # But maybe we can find 2?
                    pass

    # If we still have time, try a pure random search on 1..10000 without base scaling
    # This might find a set where sums are smooth without a common factor.
    if time.time() < deadline - 5:
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            nums = random.sample(range(1, 10001), 12)
            current_score = score_numbers(nums)
            no_improve = 0
            while no_improve < 100 and time.time() < deadline - 2:
                idx = random.randint(0, 11)
                old_val = nums[idx]
                new_val = random.randint(1, 10000)
                if new_val in nums:
                    no_improve += 1
                    continue
                nums[idx] = new_val
                test_score = score_numbers(nums)
                if test_score < current_score:
                    current_score = test_score
                    no_improve = 0
                else:
                    nums[idx] = old_val
                    no_improve += 1
            
            if current_score < best_score:
                best_score = current_score
                best_nums = nums[:]

    return best_nums

# Ensure valid output
if __name__ == "__main__":
    res = construct()
    assert len(res) == 12
    assert len(set(res)) == 12
    assert all(1 <= x <= 10000 for x in res)
    print(res)
    print(score_numbers(res))
