"""Program to minimize distinct prime factors across pairwise sums."""
import random
import math
from itertools import combinations

def _sieve_smallest_prime(limit):
    """Return list spf where spf[i] is smallest prime factor of i."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors(n, spf):
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

def _count_distinct_primes(arr, spf, threshold=float('inf')):
    """Count distinct primes dividing all pairwise sums of arr. Stop if count >= threshold."""
    prime_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                prime_set.add(p)
                if len(prime_set) >= threshold:
                    return len(prime_set)
                while n % p == 0:
                    n //= p
    return len(prime_set)

def _generate_smooth_numbers(limit, max_prime_factor=13):
    """Generate numbers <= limit with all prime factors <= max_prime_factor."""
    spf = _sieve_smallest_prime(limit)
    smooth = set()
    for i in range(1, limit + 1):
        n = i
        is_smooth = True
        while n > 1:
            p = spf[n]
            if p > max_prime_factor:
                is_smooth = False
                break
            while n % p == 0:
                n //= p
        if is_smooth:
            smooth.add(i)
    return sorted(smooth)

def construct():
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor up to 20000 (max sum of two <= 10000)
    limit = 2 * max_val
    spf = _sieve_smallest_prime(limit)
    
    # Generate pool of smooth numbers (prime factors <= 13)
    smooth_pool = _generate_smooth_numbers(max_val, max_prime_factor=13)
    
    if len(smooth_pool) < n:
        smooth_pool = list(range(1, max_val + 1))
        
    random.seed(42)
    
    best_arr = None
    best_score = float('inf')
    current_arr = None
    current_score = float('inf')
    
    # Try multiple random initializations from the smooth pool
    for _ in range(50):
        # Randomly pick n distinct numbers from smooth_pool
        if len(smooth_pool) >= n:
            arr = random.sample(smooth_pool, n)
        else:
            arr = random.sample(list(range(1, max_val+1)), n)
            
        score = _count_distinct_primes(arr, spf)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            current_arr = list(arr)
            current_score = score
            
    if best_arr is None:
        best_arr = list(range(1, n + 1))
        current_arr = list(best_arr)
        best_score = _count_distinct_primes(best_arr, spf)
        current_score = best_score

    # Local Search with Kicks
    max_iterations = 500
    no_improve_count = 0
    
    for _iter in range(max_iterations):
        improved = False
        # Shuffle the order of indices to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            old_val = current_arr[i]
            current_vals = set(current_arr)
            
            # Candidate replacements: sample from smooth pool
            candidates_to_try = [v for v in smooth_pool if v not in current_vals]
            
            if not candidates_to_try:
                continue
                
            # Sample a subset to keep it fast
            sample_size = min(50, len(candidates_to_try))
            candidates_to_try = random.sample(candidates_to_try, sample_size)
            
            for new_val in candidates_to_try:
                trial = list(current_arr)
                trial[i] = new_val
                new_score = _count_distinct_primes(trial, spf, best_score)
                
                if new_score < best_score:
                    current_score = new_score
                    current_arr = trial
                    improved = True
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_arr = list(trial)
                        
                    no_improve_count = 0
                    break # Restart inner loop with new best for this i
                    
            if improved:
                break # Restart outer loop for this i? 
                # Actually, breaking the i loop and going to next iteration is standard.
                # But we can also just continue to next i. 
                # Let's break to prioritize the improvement.
                break
                
        if not improved:
            no_improve_count += 1
            # If stuck, do a random kick
            if no_improve_count > 10:
                # Randomly change 2 elements to random smooth numbers
                idxs = random.sample(range(n), 2)
                for idx in idxs:
                    new_val = random.choice(smooth_pool)
                    # Ensure distinctness
                    attempts = 0
                    while new_val in current_arr and attempts < 10:
                        new_val = random.choice(smooth_pool)
                        attempts += 1
                    if new_val not in current_arr:
                        current_arr[idx] = new_val
                        
                # Update current score
                current_score = _count_distinct_primes(current_arr, spf, best_score)
                
                # Update best if better (unlikely)
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(current_arr)
                    
                no_improve_count = 0
                
    return best_arr
