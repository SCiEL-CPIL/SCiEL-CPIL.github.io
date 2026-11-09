import math
import time
import random
import heapq
from itertools import product

MAX_N = 10**18
DEADLINE = 138.0  # Safety margin before 150s kill

def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Generate primes up to 100000 for a larger search space
PRIMES = generate_primes(100000)
PRIME_SET = set(PRIMES)

def get_divisors_sorted(primes_list, exponents_list):
    """
    Generate divisors in sorted order.
    For moderate divisor counts, generate all and sort.
    """
    if not primes_list:
        return [1]
    
    divs = [1]
    for p, e in zip(primes_list, exponents_list):
        if e == 0:
            continue
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        new_divs = []
        for d in divs:
            for pow_p in powers:
                new_divs.append(d * pow_p)
        divs = new_divs
    
    divs.sort()
    return divs

def score_from_divs(divs):
    """Count adjacent coprime pairs in sorted divisors."""
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate_structure(primes_list, exponents_list):
    """
    Evaluate the score for a given prime exponent structure.
    Returns (score, n, num_divisors)
    """
    n = 1
    num_divs = 1
    for p, e in zip(primes_list, exponents_list):
        if e > 0:
            n *= (p ** e)
            if n > MAX_N:
                return -1, n, 0
            num_divs *= (e + 1)
    
    if n < 2:
        return -1, n, 0
    
    if num_divs > 300000:
        # Too many divisors to enumerate efficiently
        return -1, n, num_divs
    
    divs = get_divisors_sorted(primes_list, exponents_list)
    s = score_from_divs(divs)
    return s, n, len(divs)

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Start with a good initial structure (squarefree with many small primes)
    # 2. Use simulated annealing with diverse moves
    # 3. Final greedy polish
    
    # Initial: squarefree product of first k primes where product <= MAX_N
    init_primes = []
    product_val = 1
    for p in PRIMES:
        if product_val * p <= MAX_N:
            init_primes.append(p)
            product_val *= p
        else:
            break
    init_exps = [1] * len(init_primes)
    
    best_score, best_n, _ = evaluate_structure(init_primes, init_exps)
    best_primes = init_primes.copy()
    best_exps = init_exps.copy()
    
    # Simulated annealing
    temperature = 20.0
    improved = True
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < DEADLINE:
        iteration += 1
        temperature *= 0.995  # Slow cooling
        
        current_primes = best_primes.copy()
        current_exps = best_exps.copy()
        current_score = best_score
        
        current_n = 1
        for p, e in zip(current_primes, current_exps):
            if e > 0:
                current_n *= (p ** e)
        
        # Generate and evaluate candidate moves
        candidates = []
        
        # Move type 1: Change exponent of a prime
        for i, (p, e) in enumerate(zip(current_primes, current_exps)):
            # Increase exponent
            if e > 0 and current_n * p <= MAX_N:
                new_exps = current_exps.copy()
                new_exps[i] = e + 1
                candidates.append(('exp_inc', i, p, current_primes, new_exps))
            
            # Decrease exponent
            if e > 1:
                new_exps = current_exps.copy()
                new_exps[i] = e - 1
                candidates.append(('exp_dec', i, p, current_primes, new_exps))
            elif e == 1:
                # Remove prime
                new_primes = [pp for j, pp in enumerate(current_primes) if j != i]
                new_exps = [ee for j, ee in enumerate(current_exps) if j != i]
                candidates.append(('remove', i, p, new_primes, new_exps))
        
        # Move type 2: Replace a prime with another
        used_primes = set(p for p, e in zip(current_primes, current_exps) if e > 0)
        for i, p in enumerate(current_primes):
            if current_exps[i] == 0:
                continue
            max_q = MAX_N * p // current_n
            # Sample some valid replacement primes
            valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
            if valid_qs:
                if len(valid_qs) > 30:
                    sampled = random.sample(valid_qs, 30)
                else:
                    sampled = valid_qs
                for q in sampled:
                    new_primes = current_primes.copy()
                    new_primes[i] = q
                    candidates.append(('swap', i, q, new_primes, current_exps.copy()))
        
        # Move type 3: Add a new prime
        max_q = MAX_N // current_n
        valid_qs = [q for q in PRIMES if q <= max_q and q not in used_primes]
        if valid_qs:
            if len(valid_qs) > 20:
                sampled = random.sample(valid_qs, 20)
            else:
                sampled = valid_qs
            for q in sampled:
                new_primes = current_primes + [q]
                new_exps = current_exps + [1]
                candidates.append(('add', len(current_primes), q, new_primes, new_exps))
        
        # Move type 4: Swap two primes' positions (reordering doesn't matter for divisors, skip)
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        # Evaluate candidates
        found_better = False
        for move_type, idx, prime, new_primes, new_exps in candidates:
            if time.time() - start_time > DEADLINE:
                break
            
            s, n, num_divs = evaluate_structure(new_primes, new_exps)
            if s < 0:
                continue
            
            # Acceptance criteria
            if s > current_score:
                best_score = s
                best_n = n
                best_primes = new_primes
                best_exps = new_exps
                found_better = True
                no_improve_count = 0
                break
            elif s == current_score and random.random() < 0.05:
                # Accept equal with small probability
                current_primes = new_primes
                current_exps = new_exps
            elif temperature > 0.1 and random.random() < math.exp((s - current_score) / temperature):
                # Annealing acceptance
                current_primes = new_primes
                current_exps = new_exps
        
        if not found_better:
            no_improve_count += 1
            if no_improve_count > 50:
                # Try a bigger jump: reset to a different random structure
                if temperature < 1.0:
                    break
                no_improve_count = 0
                # Generate a random structure
                random_start = random.randint(5, min(20, len(PRIMES)))
                new_primes = random.sample(PRIMES[:500], random_start)
                new_exps = [1] * random_start
                # Check if valid
                n_test = 1
                for p in new_primes:
                    n_test *= p
                    if n_test > MAX_N:
                        break
                if n_test <= MAX_N and n_test >= 2:
                    s_test, n_test, _ = evaluate_structure(new_primes, new_exps)
                    if s_test > 0:
                        best_primes = new_primes
                        best_exps = new_exps
                        best_score = s_test
                        best_n = n_test
                        current_primes = best_primes
                        current_exps = best_exps
                        current_score = best_score
    
    # Final greedy polish
    # Try to improve by adding/removing/swapping one prime at a time
    polish_improved = True
    while polish_improved and time.time() - start_time < DEADLINE:
        polish_improved = False
        current_primes = best_primes.copy()
        current_exps = best_exps.copy()
        current_score = best_score
        
        current_n = 1
        for p, e in zip(current_primes, current_exps):
            if e > 0:
                current_n *= (p ** e)
        
        # Try all single-prime changes
        for i in range(len(current_primes)):
            p = current_primes[i]
            e = current_exps[i]
            
            # Try removing
            if e == 1:
                new_primes = [pp for j, pp in enumerate(current_primes) if j != i]
                new_exps = [ee for j, ee in enumerate(current_exps) if j != i]
                s, n, _ = evaluate_structure(new_primes, new_exps)
                if s > current_score:
                    best_score = s
                    best_n = n
                    best_primes = new_primes
                    best_exps = new_exps
                    polish_improved = True
                    break
            
            # Try increasing exponent
            if e > 0 and current_n * p <= MAX_N:
                new_exps = current_exps.copy()
                new_exps[i] = e + 1
                s, n, _ = evaluate_structure(current_primes, new_exps)
                if s > current_score:
                    best_score = s
                    best_n = n
                    best_exps = new_exps
                    polish_improved = True
                    break
        
        # Try adding a prime
        max_q = MAX_N // current_n
        for q in PRIMES:
            if q > max_q:
                break
            if q in set(current_primes):
                continue
            new_primes = current_primes + [q]
            new_exps = current_exps + [1]
            s, n, _ = evaluate_structure(new_primes, new_exps)
            if s > current_score:
                best_score = s
                best_n = n
                best_primes = new_primes
                best_exps = new_exps
                polish_improved = True
                break
        
        # Try swapping a prime
        used = set(current_primes)
        for i in range(len(current_primes)):
            p = current_primes[i]
            max_q = MAX_N * p // current_n
            for q in PRIMES:
                if q > max_q:
                    break
                if q in used:
                    continue
                new_primes = current_primes.copy()
                new_primes[i] = q
                s, n, _ = evaluate_structure(new_primes, current_exps)
                if s > current_score:
                    best_score = s
                    best_n = n
                    best_primes = new_primes
                    polish_improved = True
                    break
            if polish_improved:
                break
    
    return best_n
