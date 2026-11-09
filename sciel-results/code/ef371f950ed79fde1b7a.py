"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import math

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Generate primes up to 1000
    primes = _primes_upto(1000)
    
    # Start with a reasonable initial set
    # Using small primes often helps cover the initial range densely
    initial_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    current_classes = []
    current_budget = Fraction(0)
    
    for p in initial_primes:
        # Pick a random residue to start, or 0
        current_classes.append([p, 0])
        current_budget += Fraction(1, p)
        
    # Ensure budget is valid
    if current_budget > Fraction(1):
        # Remove largest primes until valid
        while current_budget > Fraction(1) and current_classes:
            current_classes.pop()
            current_budget -= Fraction(1, current_classes[-1][0]) if current_classes else 0
            # Re-calculate budget properly
            current_budget = sum(Fraction(1, p) for p, r in current_classes)

    best_L = _compute_L(current_classes)
    best_classes = [list(c) for c in current_classes]
    best_budget = current_budget

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    import random
    
    while time.time() < deadline - 5 and T > T_min:
        T *= cooling_rate
        
        # Generate a neighbor
        new_classes = [list(c) for c in best_classes]
        new_budget = best_budget
        
        move_type = random.choice(['change_residue', 'add_prime', 'remove_prime', 'swap_prime'])
        
        if move_type == 'change_residue' and new_classes:
            idx = random.randint(0, len(new_classes) - 1)
            p, r = new_classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r != r:
                new_classes[idx] = [p, new_r]
        
        elif move_type == 'add_prime':
            used_primes = set(p for p, r in new_classes)
            available = [p for p in primes if p not in used_primes]
            if available:
                p_add = random.choice(available)
                cost = Fraction(1, p_add)
                if new_budget + cost <= Fraction(1):
                    r_add = random.randint(0, p_add - 1)
                    new_classes.append([p_add, r_add])
                    new_budget += cost
        
        elif move_type == 'remove_prime' and len(new_classes) > 1:
            idx = random.randint(0, len(new_classes) - 1)
            p_rem, r_rem = new_classes.pop(idx)
            new_budget -= Fraction(1, p_rem)
        
        elif move_type == 'swap_prime' and new_classes:
            idx = random.randint(0, len(new_classes) - 1)
            p_old, r_old = new_classes[idx]
            used_primes = set(p for p, r in new_classes)
            available = [p for p in primes if p not in used_primes]
            if available:
                p_new = random.choice(available)
                cost_new = Fraction(1, p_new)
                cost_old = Fraction(1, p_old)
                if new_budget - cost_old + cost_new <= Fraction(1):
                    r_new = random.randint(0, p_new - 1)
                    new_classes[idx] = [p_new, r_new]
                    new_budget -= cost_old
                    new_budget += cost_new
        
        if new_classes == best_classes:
            continue
            
        new_L = _compute_L(new_classes)
        
        # Acceptance probability
        if new_L > best_L:
            best_L = new_L
            best_classes = new_classes
            best_budget = new_budget
        else:
            delta = new_L - best_L
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    best_L = new_L
                    best_classes = new_classes
                    best_budget = new_budget
    
    # Final local search polish
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i, (p, r) in enumerate(best_classes):
            best_r = r
            best_L_i = best_L
            for r2 in range(p):
                if r2 == r:
                    continue
                test = [list(c) for c in best_classes]
                test[i] = [p, r2]
                test_L = _compute_L(test)
                if test_L > best_L_i:
                    best_L_i = test_L
                    best_r = r2
            if best_r != r:
                best_classes[i] = [p, best_r]
                best_L = best_L_i
                improved = True
                
    return best_classes


def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _seed_classes(primes):
    seed_primes = [p for p in primes if p >= 13]
    return [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]


def _compute_L(classes):
    """Compute the largest L such that every integer 1..L is covered by some class."""
    if not classes:
        return 0
    
    # For efficiency, use a bitset or set of covered integers.
    # L is capped at 10^5, but we're mostly interested in L up to ~1000.
    max_L = 10000  # reasonable upper bound for search
    
    # Build a boolean array for coverage
    covered = [False] * (max_L + 1)
    
    for p, r in classes:
        # Cover integers n where n ≡ r (mod p), i.e., n = r, r+p, r+2p, ...
        # But n must be >= 1.
        # First integer >= 1 with n ≡ r (mod p):
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p * ((1 - first + p - 1) // p)
        
        for n in range(first, max_L + 1, p):
            covered[n] = True
    
    # Find the largest L such that covered[1..L] are all True
    L = 0
    for n in range(1, max_L + 1):
        if covered[n]:
            L = n
        else:
            break
    
    return L

# EVOLVE-BLOCK-END
