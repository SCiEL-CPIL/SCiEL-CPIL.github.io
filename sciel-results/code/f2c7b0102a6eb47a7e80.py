import math
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    LIMIT = 10**18
    
    # List of first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    def get_divisors(n):
        # Generate divisors from prime factorization
        # n is represented as list of (p, e)
        divs = [1]
        for p, e in n:
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        return sorted(divs)

    def calc_score(n_factors):
        divs = get_divisors(n_factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count

    def factors_to_n(factors):
        n = 1
        for p, e in factors:
            n *= (p ** e)
        return n

    # Initial candidate: Product of first 16 primes
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53
    # This is approx 3.25 * 10^17
    initial_factors = [(p, 1) for p in primes[:16]]
    current_factors = initial_factors[:]
    current_n = factors_to_n(current_factors)
    
    best_factors = current_factors[:]
    best_score = calc_score(current_factors)
    
    # Local search
    # We will try to adjust exponents
    # To speed up, we can precompute some things or just run the loop
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    # List of active primes (indices into primes list)
    # We might want to try adding more primes or changing exponents
    
    steps = 0
    while time.time() - start_time < DEADLINE:
        # Create a copy of current factors
        trial_factors = current_factors[:]
        
        # Choose a random action
        action = random.randint(0, 2)
        
        if action == 0:
            # Increase exponent of a random existing prime
            if not trial_factors:
                continue
            idx = random.randint(0, len(trial_factors) - 1)
            p, e = trial_factors[idx]
            # Check if we can increase
            if current_n * p <= LIMIT:
                trial_factors[idx] = (p, e + 1)
            else:
                # Try a different prime
                continue
        elif action == 1:
            # Decrease exponent of a random existing prime
            if not trial_factors:
                continue
            idx = random.randint(0, len(trial_factors) - 1)
            p, e = trial_factors[idx]
            if e > 1:
                trial_factors[idx] = (p, e - 1)
            elif e == 1:
                # Remove it
                trial_factors.pop(idx)
            else:
                continue
        else:
            # Add a new prime (if not already present) or replace a large one
            # Find smallest prime not in trial_factors
            existing_primes = {p for p, e in trial_factors}
            candidate_primes = [p for p in primes if p not in existing_primes]
            if candidate_primes:
                # Pick a small one
                p_new = min(candidate_primes)
                if current_n * p_new <= LIMIT:
                    trial_factors.append((p_new, 1))
                    # Sort by prime to keep consistent? Not strictly necessary for calculation, but good for state
                    trial_factors.sort(key=lambda x: x[0])
                else:
                    # Try to swap a large prime for this one?
                    # Or just skip
                    continue
            else:
                # All primes in list are present? Unlikely
                continue
                
        # Calculate new n
        trial_n = factors_to_n(trial_factors)
        if trial_n > LIMIT:
            continue
            
        # Calculate score
        trial_score = calc_score(trial_factors)
        
        if trial_score > best_score:
            best_score = trial_score
            best_factors = trial_factors[:]
            current_factors = trial_factors[:]
            current_n = trial_n
            # Optional: Print progress for debugging if allowed, but usually silent
            # print(f"New best: {best_score} at {trial_n}", file=sys.stderr)
            
        steps += 1
        # Small sleep to avoid CPU spinning if calc is very fast? No, calc is the bottleneck.
        
    return factors_to_n(best_factors)

# EVOLVE-BLOCK-END
