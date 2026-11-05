import time
import random
from itertools import combinations

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Primes up to 20000
PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)

def factorize(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            factors.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def count_distinct_primes(nums):
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            s = nums[i] + nums[j]
            all_primes.update(factorize(s))
    return len(all_primes)

# Generate 2,3,5,7,11-smooth numbers up to 10000
def generate_smooth_numbers(max_val, primes):
    nums = {1}
    for p in primes:
        current = list(nums)
        for x in current:
            y = x
            while y * p <= max_val:
                y *= p
                nums.add(y)
    return sorted(nums)

smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7, 11])

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # Initial candidate set: small smooth numbers
    # We want to find a set of 12 with minimal prime union.
    # Start with the smallest 12 smooth numbers.
    current_set = smooth_nums[:12]
    current_score = count_distinct_primes(current_set)
    
    best_set = current_set[:]
    best_score = current_score
    
    # Local Search
    # We will try to replace elements in the set with other smooth numbers
    # to reduce the number of distinct primes.
    
    # To speed up factorization, we can precompute primes of sums?
    # Sums are <= 20000.
    
    # Heuristic: Try to keep the set "tight" (small numbers) because
    # small sums have fewer prime factors on average?
    # Actually, large numbers might have large prime factors.
    # So staying with small smooth numbers is a good bet.
    
    # Let's try to optimize within the first 100 smooth numbers.
    candidate_pool = smooth_nums[:200]
    
    # Random restarts or local search from current best
    # We use a simple hill climbing with random restarts if stuck.
    
    def optimize(set_to_opt):
        s = set_to_opt[:]
        score = count_distinct_primes(s)
        for _ in range(5000): # Iterations per restart
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            old_val = s[i]
            # Try a random replacement from pool
            new_val = random.choice(candidate_pool)
            if new_val in s:
                continue
            s[i] = new_val
            new_score = count_distinct_primes(s)
            if new_score <= score:
                score = new_score
                # Accept
            else:
                # Revert (Hill Climbing)
                s[i] = old_val
        return s, score

    best_score = 100
    best_set = []
    
    # Try a few random initializations from the pool
    for restart in range(20):
        if time.time() > deadline:
            break
        init_set = random.sample(candidate_pool, 12)
        s, score = optimize(init_set)
        if score < best_score:
            best_score = score
            best_set = s
            if best_score <= 3: # Theoretical minimum is likely 2 or 3
                break

    # Also try the specific "small" set
    s, score = optimize(smooth_nums[:12])
    if score < best_score:
        best_score = score
        best_set = s

    return best_set
