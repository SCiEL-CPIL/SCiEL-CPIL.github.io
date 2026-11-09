"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import heapq

def _sieve(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _divisors_sorted(prime_exp_list):
    """Generate all divisors of n = prod p_i^e_i in sorted order using a heap.
    prime_exp_list: list of (prime, exponent) tuples.
    Returns sorted list of divisors.
    """
    if not prime_exp_list:
        return [1]
    
    k = len(prime_exp_list)
    # Each element in the state is a tuple (divisor_value, index_tuple)
    # index_tuple[i] is the exponent of the i-th prime in this divisor
    # We use a min-heap to generate divisors in order.
    
    # To avoid duplicates, we use a set of index tuples we've already processed.
    # However, for k up to 15 and 2^k divisors, a set of tuples is fine.
    # A better way: standard "k smallest sums" style generation.
    # Let's use the standard algorithm for generating sorted divisors.
    
    # Start with 1.
    # The next divisors are formed by multiplying by primes, but we need to avoid duplicates.
    # A common method:
    # Initialize heap with (1, 0, ...) where the state is (value, indices)
    # Actually, let's use the method where we keep track of the last prime used.
    # Or simply:
    # heap contains (value, indices_tuple)
    # We push (1, (0,)*k)
    # When we pop (val, idxs), we generate neighbors by incrementing one index,
    # provided we don't create a duplicate.
    # To avoid duplicates, we can enforce that if we increment index j,
    # then for all i > j, idxs[i] must be 0. This is a standard technique.
    
    # Let's try a simpler approach first: generate all divisors, then sort.
    # For k=15, 2^15 = 32768 divisors. Sorting 32768 integers is very fast.
    # The heap approach is O(2^k log 2^k) = O(2^k * k).
    # Generating all is O(2^k). Sorting is O(2^k log 2^k).
    # So sorting is likely faster in practice for k <= 20.
    
    divisors = [1]
    for p, e in prime_exp_list:
        new_divs = []
        power = 1
        for _ in range(e):
            power *= p
            for d in divisors:
                new_divs.append(d * power)
        divisors.extend(new_divs)
    divisors.sort()
    return divisors

def _tau_perp_from_prime_exp(prime_exp_list):
    """Compute tau_perp for n given its prime factorization as list of (p, e)."""
    divisors = _divisors_sorted(prime_exp_list)
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _product_of_prime_exp(prime_exp_list):
    """Compute product of p^e for (p, e) in prime_exp_list."""
    prod = 1
    for p, e in prime_exp_list:
        prod *= p ** e
    return prod

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    limit = 10**18
    
    # Generate a larger prime pool
    max_prime_val = 5000
    primes = _sieve(max_prime_val)
    
    # Initial solution: first 15 primes with exponent 1
    k = 15
    current_pe = [(primes[i], 1) for i in range(k)]
    current_n = _product_of_prime_exp(current_pe)
    current_score = _tau_perp_from_prime_exp(current_pe)
    
    best_pe = list(current_pe)
    best_n = current_n
    best_score = current_score
    
    # Simulated annealing
    T_start = 50.0
    T_end = 0.1
    alpha = 0.9995  # cooling rate
    
    T = T_start
    step = 0
    
    while time.time() < deadline - 10:
        if T < T_end:
            break
        
        # Generate a random move
        move_type = random.random()
        new_pe = list(current_pe)
        
        if move_type < 0.5:
            # Single swap: replace one prime
            i = random.randint(0, len(new_pe) - 1)
            # Pick a random prime not in current set
            current_prime_set = set(p for p, e in new_pe)
            available = [p for p in primes if p not in current_prime_set]
            if not available:
                T *= alpha
                continue
            new_p = random.choice(available)
            new_pe[i] = (new_p, new_pe[i][1])
            new_pe.sort(key=lambda x: x[0])
        elif move_type < 0.75:
            # Double swap: replace two primes
            if len(new_pe) < 2:
                T *= alpha
                continue
            i, j = random.sample(range(len(new_pe)), 2)
            current_prime_set = set(p for p, e in new_pe)
            available = [p for p in primes if p not in current_prime_set]
            if len(available) < 2:
                T *= alpha
                continue
            new_p1, new_p2 = random.sample(available, 2)
            new_pe[i] = (new_p1, new_pe[i][1])
            new_pe[j] = (new_p2, new_pe[j][1])
            new_pe.sort(key=lambda x: x[0])
        elif move_type < 0.9:
            # Change exponent of one prime
            i = random.randint(0, len(new_pe) - 1)
            p, e = new_pe[i]
            # Try increasing or decreasing exponent
            if random.random() < 0.5 and e < 10:
                new_e = e + 1
            elif e > 1:
                new_e = e - 1
            else:
                T *= alpha
                continue
            new_pe[i] = (p, new_e)
        else:
            # Try adding or removing a prime (change k)
            if move_type < 0.95:
                # Add a prime
                current_prime_set = set(p for p, e in new_pe)
                available = [p for p in primes if p not in current_prime_set]
                if not available:
                    T *= alpha
                    continue
                new_p = random.choice(available)
                new_pe.append((new_p, 1))
                new_pe.sort(key=lambda x: x[0])
            else:
                # Remove a prime
                if len(new_pe) <= 2:
                    T *= alpha
                    continue
                i = random.randint(0, len(new_pe) - 1)
                new_pe.pop(i)
                new_pe.sort(key=lambda x: x[0])
        
        # Check if valid (product <= limit)
        new_n = _product_of_prime_exp(new_pe)
        if new_n > limit or new_n < 2:
            T *= alpha
            continue
        
        # Compute score
        new_score = _tau_perp_from_prime_exp(new_pe)
        
        # Accept or reject based on simulated annealing
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvement
            current_pe = new_pe
            current_n = new_n
            current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_pe = list(current_pe)
        else:
            # Accept worse solution with probability exp(delta/T)
            if T > 0 and random.random() < math.exp(delta / T):
                current_pe = new_pe
                current_n = new_n
                current_score = new_score
        
        # Cool down
        T *= alpha
        step += 1
    
    # Final local search to polish
    if time.time() < deadline - 5:
        # Greedy improvement: try all single swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_pe)):
                for p in primes:
                    if any(p == bp for bp, be in best_pe):
                        continue
                    new_pe = list(best_pe)
                    new_pe[i] = (p, best_pe[i][1])
                    new_pe.sort(key=lambda x: x[0])
                    new_n = _product_of_prime_exp(new_pe)
                    if new_n > limit:
                        continue
                    new_score = _tau_perp_from_prime_exp(new_pe)
                    if new_score > best_score:
                        best_score = new_score
                        best_n = new_n
                        best_pe = new_pe
                        improved = True
                        break
                if improved:
                    break
        
        # Also try changing exponents
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(best_pe)):
                    p, e = best_pe[i]
                    # Try increasing exponent
                    if e < 10:
                        new_pe = list(best_pe)
                        new_pe[i] = (p, e + 1)
                        new_n = _product_of_prime_exp(new_pe)
                        if new_n <= limit:
                            new_score = _tau_perp_from_prime_exp(new_pe)
                            if new_score > best_score:
                                best_score = new_score
                                best_n = new_n
                                best_pe = new_pe
                                improved = True
                                break
                    # Try decreasing exponent
                    if e > 1:
                        new_pe = list(best_pe)
                        new_pe[i] = (p, e - 1)
                        new_n = _product_of_prime_exp(new_pe)
                        if new_n <= limit:
                            new_score = _tau_perp_from_prime_exp(new_pe)
                            if new_score > best_score:
                                best_score = new_score
                                best_n = new_n
                                best_pe = new_pe
                                improved = True
                                break
                if improved:
                    break
    
    return best_n
# EVOLVE-BLOCK-END
