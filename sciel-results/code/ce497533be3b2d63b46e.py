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

def _count_distinct_primes(arr, spf):
    """Count distinct primes dividing all pairwise sums of arr."""
    prime_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                prime_set.add(p)
                while n % p == 0:
                    n //= p
    return len(prime_set)

def _get_prime_counts(arr, spf):
    """
    Returns (prime_counts_dict, current_prime_set) 
    prime_counts_dict maps prime -> count of sums divisible by it
    current_prime_set is the set of primes with count > 0
    """
    counts = {}
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            factors = _prime_factors(s, spf)
            for p in factors:
                counts[p] = counts.get(p, 0) + 1
    current_primes = set(counts.keys())
    return counts, current_primes

def construct():
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor up to 20000 (max sum of two <= 10000)
    limit = 2 * max_val
    spf = _sieve_smallest_prime(limit)
    
    # Initial strategy: try to use numbers with few distinct prime factors in their sums
    # We will use a local search with efficient score updates.
    
    # Generate several starting candidates
    candidates = []
    
    # 1. Consecutive evens
    candidates.append(list(range(2, 2*n+1, 2)))
    
    # 2. Consecutive multiples of 6
    candidates.append(list(range(6, 6*n+1, 6)))
    
    # 3. Random evens
    random.seed(42)
    for _ in range(10):
        cand = random.sample(range(2, max_val+1, 2), n)
        candidates.append(cand)
        
    # 4. Powers of 2 scaled?
    # 5. Arithmetic progression with step 2, starting at random even
    for start in [10, 20, 50, 100]:
        candidates.append([start + 2*i for i in range(n)])

    best_score = float('inf')
    best_arr = None
    best_counts = None
    best_primes = None

    for cand in candidates:
        if len(set(cand)) != n or any(x < 1 or x > max_val for x in cand):
            continue
        counts, primes = _get_prime_counts(cand, spf)
        score = len(primes)
        if score < best_score:
            best_score = score
            best_arr = list(cand)
            best_counts = counts
            best_primes = primes

    if best_arr is None:
        best_arr = list(range(1, n + 1))
        best_counts, best_primes = _get_prime_counts(best_arr, spf)
        best_score = len(best_primes)

    # Local search with efficient updates
    # We replace one element at a time.
    # To avoid getting stuck, we can use a simple hill climbing with restarts or simulated annealing.
    # Given the small n, hill climbing with random restarts is good.
    
    max_restarts = 50
    max_iterations_per_restart = 5000
    
    for restart in range(max_restarts):
        if restart > 0:
            # Restart from a random perturbation of the best found so far, or a new random set
            if restart % 5 == 0:
                # New random start
                current_arr = random.sample(range(2, max_val+1, 2), n)
            else:
                # Perturb best
                current_arr = list(best_arr)
                for _ in range(3):
                    idx = random.randint(0, n-1)
                    current_arr[idx] = random.randint(1, max_val)
                    # Ensure distinct
                    while len(set(current_arr)) < n:
                        current_arr[random.randint(0, n-1)] = random.randint(1, max_val)
            
            # Ensure validity
            if any(x < 1 or x > max_val for x in current_arr) or len(set(current_arr)) < n:
                continue
                
            current_counts, current_primes = _get_prime_counts(current_arr, spf)
            current_score = len(current_primes)
        else:
            current_arr = list(best_arr)
            current_counts = dict(best_counts)
            current_primes = set(best_primes)
            current_score = best_score

        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations_per_restart:
            improved = False
            iter_count += 1
            
            # Pick a random index to replace
            i = random.randint(0, n - 1)
            old_val = current_arr[i]
            
            # Generate a few candidate replacements
            # Heuristic: try values close to old_val, or values that make sums smooth
            # For efficiency, we'll just try random even numbers first, then specific ones
            
            candidates_repl = set()
            
            # Random evens
            for _ in range(20):
                candidates_repl.add(random.randint(2, max_val, 2))
            
            # Neighbors
            for delta in range(-20, 21, 2):
                v = old_val + delta
                if 1 <= v <= max_val:
                    candidates_repl.add(v)
            
            # Multiples of small primes (2, 3, 6)
            for p in [2, 3, 6]:
                base = (old_val // p) * p
                for mult in range(base - 10*p, base + 10*p, p):
                    if 1 <= mult <= max_val:
                        candidates_repl.add(mult)
            
            for new_val in candidates_repl:
                if new_val == old_val or new_val in current_arr:
                    continue
                
                # Calculate delta score
                # Sums involving i change: current_arr[i] + current_arr[j] -> new_val + current_arr[j]
                # Other sums remain same.
                
                # We need to update counts for primes in old sums and new sums.
                # To avoid modifying the global state until we accept, we can simulate.
                # Since n=12, there are 11 sums affected.
                
                new_counts = dict(current_counts)
                new_primes = set(current_primes)
                
                # Decrement counts for old sums
                for j in range(n):
                    if j == i:
                        continue
                    old_sum = old_val + current_arr[j]
                    old_factors = _prime_factors(old_sum, spf)
                    for p in old_factors:
                        new_counts[p] -= 1
                        if new_counts[p] == 0:
                            new_primes.discard(p)
                            # Optionally delete from dict to keep it clean, but not strictly necessary if we check >0
                            
                # Increment counts for new sums
                for j in range(n):
                    if j == i:
                        continue
                    new_sum = new_val + current_arr[j]
                    new_factors = _prime_factors(new_sum, spf)
                    for p in new_factors:
                        if p not in new_counts:
                            new_counts[p] = 0
                        new_counts[p] += 1
                        if new_counts[p] == 1:
                            new_primes.add(p)
                
                new_score = len(new_primes)
                
                if new_score < current_score:
                    # Accept
                    current_arr[i] = new_val
                    current_counts = new_counts
                    current_primes = new_primes
                    current_score = new_score
                    improved = True
                    break # Restart inner loop? Or continue? 
                    # Breaking allows us to pick a new random index immediately, which is good for exploration.
                    
        # End of local search for this restart
        if current_score < best_score:
            best_score = current_score
            best_arr = list(current_arr)
            best_counts = current_counts
            best_primes = current_primes
            
    return best_arr
