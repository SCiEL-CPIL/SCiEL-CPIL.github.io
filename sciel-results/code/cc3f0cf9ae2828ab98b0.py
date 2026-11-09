import math
import time
import random

def _sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0: is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

# We need primes up to a reasonable limit. 
# Max number of distinct primes for n <= 10^18 is 15.
_PRIMES = _sieve(100)
CANDIDATE_PRIMES = _PRIMES[:15]
N_PRIMES = len(CANDIDATE_PRIMES)

def _get_divisors(exponents, primes):
    """
    Generate all divisors of n = prod(p_i ^ e_i)
    Returns a sorted list of divisors.
    """
    divs = [1]
    for i, p in enumerate(primes):
        e = exponents[i]
        if e == 0:
            continue
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        new_divs = []
        for d in divs:
            for pw in powers:
                new_divs.append(d * pw)
        divs = new_divs
    divs.sort()
    return divs

def _score_from_divs(divs):
    """
    Count adjacent coprime pairs in a sorted list of divisors.
    """
    score = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            score += 1
    return score

def _score(exponents, primes):
    """
    Calculate score for given exponents.
    Returns 0 if n > 10^18 or n < 2.
    """
    n = 1
    for i, p in enumerate(primes):
        if exponents[i] > 0:
            # Check overflow
            if n > (10**18) // (p ** exponents[i]):
                return 0
            n *= p ** exponents[i]
    
    if n < 2 or n > 10**18:
        return 0
        
    divs = _get_divisors(exponents, primes)
    return _score_from_divs(divs)

def _calculate_n(exponents, primes):
    n = 1
    for i, p in enumerate(primes):
        if exponents[i] > 0:
            n *= p ** exponents[i]
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial population: diverse set of exponent vectors
    # 1. Greedy max divisors
    # 2. All ones
    # 3. Random
    # 4. Power of 2, 3, etc.
    
    population = []
    
    # Candidate 1: Greedy fill to maximize tau(n)
    exps1 = [0] * N_PRIMES
    n_curr = 1
    for i, p in enumerate(CANDIDATE_PRIMES):
        if n_curr * p <= 10**18:
            exps1[i] = 1
            n_curr *= p
    population.append(exps1[:])
    
    # Candidate 2: Maximize exponent of 2
    exps2 = [0] * N_PRIMES
    exps2[0] = 59 # 2^59 < 10^18
    population.append(exps2[:])
    
    # Candidate 3: Random
    for _ in range(20):
        exps = [0] * N_PRIMES
        n_curr = 1
        for i in range(N_PRIMES):
            # Try random small exponents
            max_e = 1
            while n_curr * (CANDIDATE_PRIMES[i] ** (max_e + 1)) <= 10**18:
                max_e += 1
            if max_e > 0:
                e = random.randint(0, max_e)
                exps[i] = e
                n_curr *= CANDIDATE_PRIMES[i] ** e
        population.append(exps[:])
        
    # Evaluate initial population
    scored_population = []
    for exps in population:
        score = _score(exps, CANDIDATE_PRIMES)
        if score > 0:
            scored_population.append((score, exps))
            
    if not scored_population:
        return 2
        
    scored_population.sort(key=lambda x: x[0], reverse=True)
    best_score, best_exps = scored_population[0]
    best_n = _calculate_n(best_exps, CANDIDATE_PRIMES)
    
    # Genetic Algorithm / Population-based search
    pop_size = 50
    # Fill population with mutations of the best
    for _ in range(pop_size - len(scored_population)):
        parent = random.choice(scored_population)[1]
        child = parent[:]
        # Mutation: change 1-2 exponents
        for _ in range(random.randint(1, 2)):
            i = random.randint(0, N_PRIMES - 1)
            delta = random.choice([-1, 1])
            child[i] += delta
            if child[i] < 0:
                child[i] = 0
        scored_population.append((0, child[:]))
        
    # Main loop
    while time.time() < deadline:
        # Selection: keep top half
        scored_population.sort(key=lambda x: x[0], reverse=True)
        scored_population = scored_population[:pop_size//2]
        
        # Crossover and Mutation to create new generation
        new_population = []
        while len(new_population) < pop_size:
            # Pick two parents
            p1 = random.choice(scored_population)[1]
            p2 = random.choice(scored_population)[1]
            
            # Crossover: mix exponents
            child = [0] * N_PRIMES
            for i in range(N_PRIMES):
                if random.random() < 0.5:
                    child[i] = p1[i]
                else:
                    child[i] = p2[i]
            
            # Mutation
            if random.random() < 0.7:
                i = random.randint(0, N_PRIMES - 1)
                delta = random.choice([-1, 1])
                child[i] += delta
                if child[i] < 0:
                    child[i] = 0
            
            new_population.append(child[:])
            
        # Evaluate new population
        for i, exps in enumerate(new_population):
            score = _score(exps, CANDIDATE_PRIMES)
            new_population[i] = (score, exps)
            
        # Combine and select
        scored_population = scored_population + new_population
        scored_population.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored_population[0][0] > best_score:
            best_score = scored_population[0][0]
            best_exps = scored_population[0][1]
            best_n = _calculate_n(best_exps, CANDIDATE_PRIMES)
            
        # Keep top pop_size
        scored_population = scored_population[:pop_size]
        
        # Early stop if we hit a theoretical limit? 
        # Theoretical max is hard to know, but if we find a score that seems very high, we can stop.
        # For now, just run until deadline.

    # Final check: ensure best_n is valid
    if best_n < 2 or best_n > 10**18:
        best_n = 2
        best_score = 1
        
    return best_n

# EVOLVE-BLOCK-END
