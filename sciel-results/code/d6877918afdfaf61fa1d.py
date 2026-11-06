import math
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2):
    """
    Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint.
    They are disjoint iff (r1 - r2) is not divisible by gcd(m1, m2).
    """
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _find_residue(m_new, existing_classes):
    """
    Try to find a residue r in [0, m_new) that is disjoint from all existing classes.
    Returns the residue if found, else None.
    """
    for r in range(m_new):
        valid = True
        for r_ex, m_ex in existing_classes:
            if not _is_disjoint(r, m_new, r_ex, m_ex):
                valid = False
                break
        if valid:
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # All candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    
    # We will try multiple ordering strategies and keep the best result
    best_set = []
    
    # Strategy 1: Sort by modulus value (ascending)
    # Strategy 2: Sort by number of divisors (descending) - favors composite numbers
    # Strategy 3: Sort by "smoothness" or prime factor count
    # Strategy 4: Randomized restarts (limited by time)
    
    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
        
    def num_prime_factors(n):
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

    # Define several ordering keys
    orderings = [
        lambda m: m,  # Ascending value
        lambda m: -num_divisors(m),  # Descending divisor count
        lambda m: num_prime_factors(m),  # Ascending prime factor count
        lambda m: -m,  # Descending value
        lambda m: (num_prime_factors(m), m), # Then value
    ]
    
    # Also add a few random permutations if time permits
    import random
    random.seed(42)
    for _ in range(5):
        random_moduli = moduli[:]
        random.shuffle(random_moduli)
        orderings.append(lambda m, rm=random_moduli: rm.index(m)) # This is inefficient, let's just store the list
        
    # Let's restructure to just iterate over pre-sorted lists
    sorted_lists = []
    for key in orderings:
        sorted_lists.append(sorted(moduli, key=key))
    
    # Add some random lists
    for _ in range(10):
        random_moduli = moduli[:]
        random.shuffle(random_moduli)
        sorted_lists.append(random_moduli)
        
    for order in sorted_lists:
        if time.time() > deadline:
            break
            
        current_set = []
        for m in order:
            # Check if m is already used (it shouldn't be if we process each modulus once per list)
            # But we need to ensure distinct moduli in the final set.
            # Since we iterate through each modulus in 'order' once, and 'order' contains unique moduli,
            # we just need to pick a residue.
            
            # Optimization: Skip if we already have a class with this modulus? 
            # No, we are picking ONE class per modulus at most.
            # The problem says "distinct moduli". So we can pick at most one class per modulus.
            
            r = _find_residue(m, current_set)
            if r is not None:
                current_set.append((r, m))
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # If we found a very large set, we might stop early or continue to see if we can improve
        # 17 is the known good number. 18 is the theoretical max?
        # Let's just run all quick ones.
        
    # Final check to ensure we return a list of lists [residue, modulus]
    return [[r, m] for r, m in best_set]

# EVOLVE-BLOCK-END
