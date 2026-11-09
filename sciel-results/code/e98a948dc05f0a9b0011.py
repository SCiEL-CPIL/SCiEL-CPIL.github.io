import math
import time

MAX_N = 10**18

def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Generate a pool of primes for search
PRIMES = generate_primes(1000)

def get_divisors(primes_list):
    """
    Generate all divisors of the product of primes in primes_list.
    Assumes primes are distinct.
    """
    divs = [1]
    for p in primes_list:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * p)
        divs = new_divs
    return divs

def score_from_divs(divs):
    """
    Calculate the number of adjacent coprime pairs in the sorted list of divisors.
    """
    divs.sort()
    count = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            count += 1
    return count

def evaluate_set(primes_set):
    """
    Evaluate a set of primes. Returns (score, n).
    """
    n = 1
    for p in primes_set:
        n *= p
        if n > MAX_N:
            return -1, n
    divs = get_divisors(primes_set)
    score = score_from_divs(divs)
    return score, n

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Start with the first 15 primes
    # Product of first 15 primes is ~6.1 * 10^17
    initial_primes = PRIMES[:15]
    best_score, best_n = evaluate_set(initial_primes)
    best_primes = initial_primes[:]
    
    if best_score == -1:
        # Fallback if initial is too large (should not happen for 15 primes)
        best_primes = PRIMES[:14]
        best_score, best_n = evaluate_set(best_primes)
        
    # Local Search
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        current_primes = best_primes[:]
        
        # Try swapping a prime with another from the pool
        # We restrict the pool to primes that keep the product <= MAX_N
        # To be efficient, we only try swapping with primes that are "close" in magnitude or from a pre-filtered list
        # For now, we try all primes in PRIMES, but break early if product exceeds MAX_N
        
        for i in range(len(current_primes)):
            p_in = current_primes[i]
            # Try replacing p_in with p_out
            for j, p_out in enumerate(PRIMES):
                if p_out == p_in:
                    continue
                if p_out in current_primes:
                    continue
                
                # Quick check: if p_out is much larger, product might exceed MAX_N
                # We can estimate: n_new = n_old * p_out / p_in
                # But we don't have n_old handy without recalculating. 
                # Let's just calculate.
                
                new_primes = current_primes[:]
                new_primes[i] = p_out
                s, n = evaluate_set(new_primes)
                
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_primes = new_primes
                    current_primes = new_primes
                    improved = True
                    break
            if improved:
                break
        
        # Try adding a prime (if product allows)
        if not improved:
            for p_out in PRIMES:
                if p_out in current_primes:
                    continue
                new_primes = current_primes + [p_out]
                s, n = evaluate_set(new_primes)
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_primes = new_primes
                    improved = True
                    break
        
        # Try removing a prime (usually decreases score, but might allow adding two others? 
        # Since we are maximizing, removal is unlikely to help unless it enables a better swap later, 
        # but local search usually handles this via swaps. We skip removal to save time.)
        
    return best_n

# Run construct to ensure it's ready
# Note: The evaluator calls construct()
# We do not call it here to avoid double execution time if the evaluator imports and calls.
# But the prompt says "Keep the function names and signatures".
# The current program calls construct() at the end. I will do the same.
construct()
