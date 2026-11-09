import math
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 145.0
    
    # Candidate primes: first 500 primes is a good balance
    # We can generate them quickly
    def gen_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    # We need primes up to a certain limit. 
    # If we use 15 primes, the largest one can be around 10^18 / (product of first 14)
    # Product of first 14 primes is ~ 1.3e17. So 15th can be up to ~7.
    # Wait, primorial(15) is ~ 6e17.
    # If we swap small primes for large ones, we can have fewer primes but larger.
    # Let's use a pool of first 1000 primes.
    primes_pool = gen_primes(10000) # Primes up to 10000
    
    # Initial state: Primorial of first 15 primes
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    initial_primes = primes_pool[:15]
    current_primes = initial_primes[:]
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    def score_n(n):
        if n < 2:
            return 0
        # Generate divisors
        # Factorization is known from current_primes if we pass it, 
        # but score_n should be standalone or we pass factors.
        # For speed, we will pass the list of primes.
        pass

    def evaluate(primes_list):
        # Calculate n
        n = 1
        for p in primes_list:
            n *= p
        if n > 10**18:
            return -1, n
        
        # Generate divisors
        divisors = [1]
        for p in primes_list:
            new_divs = []
            for d in divisors:
                new_divs.append(d * p)
            divisors.extend(new_divs)
            
        divisors.sort()
        
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, n

    # Initial evaluation
    current_score, current_n = evaluate(current_primes)
    best_score = current_score
    best_primes = current_primes[:]
    best_n = current_n

    # Simulated Annealing
    T = 100.0
    T_min = 0.01
    alpha = 0.999
    # We will run until time runs out or T < T_min
    # Estimate iterations: 100,000 iterations might take too long if evaluation is slow.
    # Evaluation of 32768 divisors: ~10ms?
    # 100,000 * 10ms = 1000s. Too slow.
    # We need faster evaluation or fewer iterations.
    # Optimization: 
    # 1. Use a smaller prime pool.
    # 2. Use a smarter move generation.
    # 3. Maybe use a heuristic score that is cheaper?
    #    Heuristic: Number of primes? No, not directly correlated.
    #    Let's try to speed up divisor generation.
    #    Actually, 32768 divisors is not huge. 
    #    Python loop over 32k items is fast (~1ms).
    #    Sorting 32k items is ~5ms.
    #    GCD 32k times is ~5ms.
    #    Total ~10-15ms per evaluation.
    #    In 140s, we can do ~10,000 evaluations.
    #    This is enough for SA if we are smart.

    random.seed(42)
    
    # Precompute log primes for product check?
    # Just use integer multiplication and check > 10**18.
    
    LIMIT = 10**18
    
    # To speed up, we can restrict the search to "square-free" numbers.
    # Moves:
    # 1. Swap p_i with p_j
    # 2. Add p_j
    # 3. Remove p_i
    
    # We maintain current_n to avoid recalculating from scratch on swap/remove.
    # For add, we just multiply.
    
    # Let's use a simpler Local Search first, then SA if time permits.
    # Hill Climbing:
    # Try all swaps, pick the best improvement.
    # This is O(K^2) per step, where K is number of primes (~15).
    # 15^2 = 225 evaluations per step.
    # 10,000 steps * 225 = 2.2M evaluations. Too many.
    
    # Randomized Hill Climbing:
    # Pick random swap, if better, accept.
    # This is 1 evaluation per step.
    # 10,000 steps * 1 = 10,000 evaluations. Feasible.
    
    # Let's do Randomized Hill Climbing with Restarts.
    
    best_global_score = best_score
    best_global_n = best_n
    best_global_primes = best_primes[:]
    
    num_restarts = 20
    steps_per_restart = 500
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        if r == 0:
            current_primes = best_global_primes[:]
            current_score = best_global_score
            current_n = best_global_n
        else:
            # Random restart: Pick random subset of primes from pool
            # Size between 10 and 15
            k = random.randint(10, 15)
            # Ensure product <= 10^18
            # Greedy pick random primes, check product
            cand = []
            prod = 1
            attempts = 0
            while len(cand) < k and attempts < 100:
                p = random.choice(primes_pool)
                if p not in cand:
                    if prod * p <= LIMIT:
                        cand.append(p)
                        prod *= p
                    attempts += 1
            if len(cand) == 0:
                continue
            current_primes = cand
            current_score, current_n = evaluate(current_primes)
            if current_score < 0:
                continue
        
        # Local Search
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick a random move
            move_type = random.randint(0, 2)
            new_primes = current_primes[:]
            new_n = current_n
            valid = True
            
            if move_type == 0:
                # Swap
                if len(current_primes) > 1:
                    i = random.randint(0, len(current_primes)-1)
                    # Pick a new prime from pool
                    j_prime = random.choice(primes_pool)
                    if j_prime not in current_primes:
                        old_p = current_primes[i]
                        # Check product
                        # new_n = current_n // old_p * j_prime
                        # Use integer division carefully
                        if current_n % old_p == 0:
                            candidate_n = (current_n // old_p) * j_prime
                            if candidate_n <= LIMIT and candidate_n >= 2:
                                new_primes[i] = j_prime
                                new_n = candidate_n
                            else:
                                valid = False
                        else:
                            valid = False
                    else:
                        valid = False
            elif move_type == 1:
                # Add
                j_prime = random.choice(primes_pool)
                if j_prime not in current_primes:
                    candidate_n = current_n * j_prime
                    if candidate_n <= LIMIT:
                        new_primes.append(j_prime)
                        new_n = candidate_n
                    else:
                        valid = False
                else:
                    valid = False
            else:
                # Remove
                if len(current_primes) > 1:
                    i = random.randint(0, len(current_primes)-1)
                    old_p = current_primes[i]
                    new_primes.pop(i)
                    new_n = current_n // old_p
                    if new_n < 2:
                        valid = False
                else:
                    valid = False
            
            if not valid:
                continue
                
            new_score, _ = evaluate(new_primes)
            
            if new_score > current_score:
                current_score = new_score
                current_primes = new_primes
                current_n = new_n
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_n = current_n
                    best_global_primes = current_primes[:]
                    
    return best_global_n
