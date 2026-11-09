import time
import random
import math

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.add(d)
            n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes(s))
    return all_primes

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    max_val = 10000
    target_size = 12
    
    # Precompute prime factors for all possible sums (2 to 20000)
    sum_primes_cache = [None] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        sum_primes_cache[s] = get_primes(s)
    
    def greedy_construct(start_val, seed):
        """Build a set greedily, choosing each new element to minimize new primes."""
        rng = random.Random(seed)
        current_set = [start_val]
        current_primes = set()
        
        for _ in range(target_size - 1):
            best_candidate = None
            best_new_primes_count = float('inf')
            best_resulting_primes = None
            
            # Try a sample of candidates to keep it fast
            candidates = rng.sample(range(1, max_val + 1), 200)
            
            for candidate in candidates:
                if candidate in current_set:
                    continue
                
                # Calculate new primes introduced
                new_primes = set()
                for existing in current_set:
                    s = candidate + existing
                    if s <= 2 * max_val:
                        new_primes.update(sum_primes_cache[s])
                
                total_primes = current_primes | new_primes
                new_count = len(total_primes - current_primes)
                
                if new_count < best_new_primes_count:
                    best_new_primes_count = new_count
                    best_candidate = candidate
                    best_resulting_primes = total_primes
            
            if best_candidate is not None:
                current_set.append(best_candidate)
                current_primes = best_resulting_primes
        
        return current_set, current_primes
    
    best_set = None
    best_prime_count = float('inf')
    
    # Try multiple random seeds and starting values
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        
        start_val = random.Random(seed).randint(1, max_val)
        result_set, result_primes = greedy_construct(start_val, seed)
        
        # Verify validity
        if len(set(result_set)) == target_size and all(1 <= x <= max_val for x in result_set):
            actual_primes = evaluate(result_set)
            if len(actual_primes) < best_prime_count:
                best_prime_count = len(actual_primes)
                best_set = result_set[:]
                
                # If we've found a very good solution, we can stop early
                if best_prime_count <= 4:
                    break
    
    # If greedy didn't find anything good, try a fallback
    if best_set is None:
        best_set = random.sample(range(1, max_val + 1), target_size)
        best_prime_count = len(evaluate(best_set))
    
    # Local search improvement: try swapping one element with another
    if time.time() < deadline - 10:
        current_set = best_set[:]
        current_primes = evaluate(current_set)
        current_count = len(current_primes)
        
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            
            idx = random.randint(0, target_size - 1)
            old_val = current_set[idx]
            new_val = random.randint(1, max_val)
            
            if new_val == old_val:
                continue
            if new_val in current_set and current_set.index(new_val) != idx:
                continue
            
            new_set = current_set[:]
            new_set[idx] = new_val
            new_primes = evaluate(new_set)
            new_count = len(new_primes)
            
            if new_count < current_count:
                current_set = new_set
                current_primes = new_primes
                current_count = new_count
                
                if current_count < best_prime_count:
                    best_prime_count = current_count
                    best_set = current_set[:]
    
    return best_set
