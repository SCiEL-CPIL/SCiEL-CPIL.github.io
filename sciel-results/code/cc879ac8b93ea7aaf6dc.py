"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

# Precompute smallest prime factor (SPF) for fast factorization
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def _get_prime_factors(n):
    """Return set of prime factors of n using precomputed SPF."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _calc_cost(vals):
    """Calculate the number of distinct prime factors for all pairwise sums."""
    primes = set()
    n = len(vals)
    for i in range(n):
        for j in range(i+1, n):
            primes.update(_get_prime_factors(vals[i] + vals[j]))
    return len(primes), primes

def _incremental_cost_update(current_vals, current_primes, old_idx, new_val):
    """
    Efficiently update the prime set when replacing current_vals[old_idx] with new_val.
    Returns (new_prime_count, new_prime_set).
    """
    # This is a helper to think about, but for simplicity in SA, 
    # we might just recalculate if the set is small enough or use a delta approach.
    # Given N=12, full recalculation is 66 factorizations.
    # 66 * ~5 ops = 330 ops. This is very fast in Python if factorization is O(1) lookup + loop.
    # Let's stick to full recalculation for simplicity and correctness first, 
    # as 60s allows for ~100,000 to 1,000,000 iterations.
    pass

def construct():
    """
    Find 12 distinct positive integers <= 10000 minimizing the number of distinct primes
    dividing sums of distinct-index pairs.
    Uses Simulated Annealing on the full space of 12-tuples.
    """
    N = 12
    MAX_VAL = 10000
    
    # Initialization: 
    # Try to start with a structured set to give SA a good starting point.
    # Strategy: Pick a modulus M with few prime factors (e.g., 60 = 2*3*5).
    # Pick x_i such that x_i % M is constant, making all sums divisible by M.
    # Then the "remainder" primes are from (x_i+x_j)/M.
    # Let's try M=60, C=30 (so sums are divisible by 60? No, 30+30=60. Yes.)
    # x_i = 30 + 60*k_i.
    # Distinct k_i.
    
    best_set = None
    best_cost = float('inf')
    
    # Generate a few diverse initial states
    initial_states = []
    
    # State 1: Arithmetic progression with step 60, start 30
    # 30, 90, 150, ...
    # Sums: 120, 180, ... all divisible by 60.
    # Primes of 60: 2,3,5.
    # Quotients: 2, 3, 4... 
    # This is a good start.
    start_val = 30
    step = 60
    init1 = [start_val + step * i for i in range(N)]
    if all(x <= MAX_VAL for x in init1):
        initial_states.append(init1)
        
    # State 2: Random small numbers
    random.seed(42)
    init2 = random.sample(range(1, 200), N)
    initial_states.append(init2)
    
    # State 3: Random large numbers with step 60
    start_val = 30
    step = 60
    # Pick random k's
    k_vals = random.sample(range(0, (MAX_VAL - start_val)//step), N)
    init3 = sorted([start_val + step * k for k in k_vals])
    initial_states.append(init3)

    for init in initial_states:
        current_set = init[:]
        current_cost, current_primes = _calc_cost(current_set)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_set = current_set[:]
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        iterations = 5000 # Adjust based on time budget
        
        for _ in range(iterations):
            # Generate a move
            # 1. Replace one element
            idx = random.randint(0, N - 1)
            old_val = current_set[idx]
            
            # Generate new value
            # Bias towards values that keep sums "smooth"?
            # Just random for now, or local perturbation.
            if random.random() < 0.5:
                # Local perturbation
                new_val = old_val + random.randint(-100, 100)
            else:
                # Global random
                new_val = random.randint(1, MAX_VAL)
            
            # Ensure bounds
            if new_val < 1 or new_val > MAX_VAL:
                continue
            
            # Ensure distinctness
            if new_val in current_set and new_val != old_val:
                continue
            if new_val == old_val:
                continue

            # Apply move
            current_set[idx] = new_val
            new_cost, new_primes = _calc_cost(current_set)
            
            # Acceptance
            if new_cost < current_cost:
                current_cost = new_cost
                current_primes = new_primes
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = current_set[:]
            else:
                # Reject with probability
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                    current_primes = new_primes
                else:
                    # Revert
                    current_set[idx] = old_val
            
            T *= cooling_rate
            if T < T_min:
                break
                
    return best_set
# EVOLVE-BLOCK-END
