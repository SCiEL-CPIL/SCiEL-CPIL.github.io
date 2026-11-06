import time
import random

def is_valid(seq):
    # Check if all consecutive block products are distinct
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def get_primes_composites(max_val):
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    composites = [i for i in range(2, max_val + 1) if not is_prime[i]]
    return primes, composites

def count_prime_factors(n):
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes, composites = get_primes_composites(128)
    base_seq = primes
    
    # Sort composites by number of prime factors (fewer is easier to place?)
    # Or just randomize. Let's try sorting by value or factor count.
    # Composites with fewer factors are less likely to collide with blocks of primes.
    composites_sorted = sorted(composites, key=lambda x: count_prime_factors(x))
    
    best_seq = list(base_seq)
    best_len = len(base_seq)
    
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        current_seq = list(base_seq)
        # Shuffle composites for this restart
        comps = list(composites)
        rng.shuffle(comps)
        
        for c in comps:
            if time.time() > deadline:
                break
            # Try inserting c at every position
            valid_positions = []
            n = len(current_seq)
            for i in range(n + 1):
                # Insert c at i
                new_seq = current_seq[:i] + [c] + current_seq[i:]
                # Check if new_seq is strictly increasing
                # Since we insert c, we need current_seq[i-1] < c < current_seq[i]
                if i > 0 and current_seq[i-1] >= c:
                    continue
                if i < n and current_seq[i] <= c:
                    continue
                # Check validity
                if is_valid(new_seq):
                    valid_positions.append(i)
            
            if valid_positions:
                # Pick a random valid position to diversify
                pos = rng.choice(valid_positions)
                current_seq = current_seq[:pos] + [c] + current_seq[pos:]
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            
    return best_seq
