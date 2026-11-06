"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Uses best-first search over prime exponent configurations with swap moves.
"""
import math
import time
import heapq


def _tau_perp_from_factorization(primes, exponents):
    """Compute tau_perp(n) given its prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        p_powers = [1]
        for _ in range(e):
            p_powers.append(p_powers[-1] * p)
        for d in divisors:
            for pp in p_powers:
                new_divs.append(d * pp)
        divisors = new_divs
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count


def _generate_primes(limit):
    """Generate all primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Generate enough primes (we'll need primes up to maybe ~500)
    all_primes = _generate_primes(1000)
    
    best_n = 1
    best_score = 0
    
    def try_candidate(primes_list, exponents_list):
        nonlocal best_n, best_score
        n = 1
        for p, e in zip(primes_list, exponents_list):
            n *= p ** e
        if 2 <= n <= 10**18:
            score = _tau_perp_from_factorization(primes_list, exponents_list)
            if score > best_score:
                best_score = score
                best_n = n
            return score
        return -1
    
    # Initial candidate: product of first 15 primes
    initial_primes = all_primes[:15]
    initial_exps = [1] * 15
    try_candidate(initial_primes, initial_exps)
    
    # Best-first search over (prime_set, exponent_vector) configurations
    # State: (negative_score, prime_indices, exponents, n_value)
    # We use prime indices into all_primes for compact representation
    
    visited = set()
    
    def state_key(prime_indices, exponents):
        return (tuple(prime_indices), tuple(exponents))
    
    # Priority queue: (negative_score, counter, prime_indices, exponents)
    # counter is for tie-breaking
    counter = 0
    pq = []
    
    # Start with the initial configuration
    init_key = state_key(initial_primes, initial_exps)
    visited.add(init_key)
    init_n = 1
    for p, e in zip(initial_primes, initial_exps):
        init_n *= p ** e
    heapq.heappush(pq, (-best_score, counter, list(range(15)), [1]*15))
    counter += 1
    
    max_steps = 50000  # Safety limit on number of states explored
    
    steps = 0
    while pq and time.time() < deadline and steps < max_steps:
        neg_score, _, prime_indices, exponents = heapq.heappop(pq)
        steps += 1
        
        # Get current n value
        n_val = 1
        for idx, e in zip(prime_indices, exponents):
            n_val *= all_primes[idx] ** e
        
        if n_val > 10**18 or n_val < 2:
            continue
        
        current_score = -neg_score
        
        # Generate neighbors via swap moves:
        # 1. Increment exponent of one prime (if it fits)
        # 2. Decrement exponent of one prime and increment another
        # 3. Swap a prime with a larger one (keep exponents same)
        
        neighbors = []
        
        # Move type 1: increment exponent of prime at position i
        for i in range(len(prime_indices)):
            new_exps = exponents[:]
            new_exps[i] += 1
            new_n = n_val * all_primes[prime_indices[i]]
            if new_n <= 10**18:
                key = state_key(prime_indices, new_exps)
                if key not in visited:
                    visited.add(key)
                    neighbors.append((prime_indices[:], new_exps, new_n))
        
        # Move type 2: decrement exponent of prime i, increment exponent of prime j
        for i in range(len(prime_indices)):
            if exponents[i] <= 1:
                continue
            for j in range(len(prime_indices)):
                if i == j:
                    continue
                new_exps = exponents[:]
                new_exps[i] -= 1
                new_exps[j] += 1
                new_n = n_val // all_primes[prime_indices[i]] * all_primes[prime_indices[j]]
                if 2 <= new_n <= 10**18:
                    key = state_key(prime_indices, new_exps)
                    if key not in visited:
                        visited.add(key)
                        neighbors.append((prime_indices[:], new_exps, new_n))
        
        # Move type 3: swap prime at position i with a larger prime
        for i in range(len(prime_indices)):
            current_prime_idx = prime_indices[i]
            # Try swapping with primes at higher indices
            for new_idx in range(current_prime_idx + 1, min(current_prime_idx + 20, len(all_primes))):
                new_prime = all_primes[new_idx]
                old_prime = all_primes[current_prime_idx]
                e = exponents[i]
                new_n = n_val // (old_prime ** e) * (new_prime ** e)
                if 2 <= new_n <= 10**18:
                    new_prime_indices = prime_indices[:]
                    new_prime_indices[i] = new_idx
                    key = state_key(new_prime_indices, exponents)
                    if key not in visited:
                        visited.add(key)
                        neighbors.append((new_prime_indices, exponents[:], new_n))
        
        # Evaluate neighbors and add to priority queue
        for new_pidx, new_exps, new_n in neighbors:
            if time.time() > deadline:
                break
            score = try_candidate([all_primes[i] for i in new_pidx], new_exps)
            if score > 0:
                heapq.heappush(pq, (-score, counter, new_pidx, new_exps))
                counter += 1
    
    return best_n
