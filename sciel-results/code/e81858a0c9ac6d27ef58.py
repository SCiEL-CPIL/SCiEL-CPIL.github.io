"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _primes_upto(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _make_mask(p, r, cap):
    """Generate bitset mask for residue r mod p up to cap."""
    mask = 0
    start = r if r >= 1 else p
    if start > cap:
        return 0
    for n in range(start, cap + 1, p):
        mask |= (1 << n)
    return mask

def _compute_L_fast(classes, cap=100000):
    """Faster L computation using bitsets."""
    if not classes:
        return 0
    
    coverage = 0
    for p, r in classes:
        coverage |= _make_mask(p, r, cap)
        
    # Find first 0 bit >= 1
    not_cov = ~coverage
    not_cov &= ~1 # Clear bit 0
    
    # In Python, ~coverage is negative. 
    # not_cov & -not_cov gives the lowest set bit.
    lowest_zero = (not_cov & -not_cov).bit_length() - 1
    
    if lowest_zero > cap:
        return cap
    else:
        return lowest_zero - 1

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    all_primes = _primes_upto(1000)
    primes_asc = sorted(all_primes)
    
    # Initial states
    initial_states = []
    
    # State 1: {2, 3, 7, 53}
    c1 = [[2, 0], [3, 0], [7, 0], [53, 0]]
    cost1 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,53)
    if cost1 <= 1: initial_states.append(c1)
    
    # State 2: {2, 3, 11}
    c2 = [[2, 0], [3, 0], [11, 0]]
    cost2 = Fraction(1,2) + Fraction(1,3) + Fraction(1,11)
    if cost2 <= 1: initial_states.append(c2)
    
    # State 3: {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
    primes_s3 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    c3 = [[p, 0] for p in primes_s3]
    cost3 = sum(Fraction(1, p) for p in primes_s3)
    if cost3 <= 1: initial_states.append(c3)
    
    # State 4: Primes >= 13
    primes_s4 = [p for p in primes_asc if p >= 13]
    cost4 = sum(Fraction(1, p) for p in primes_s4)
    if cost4 <= 1:
        c4 = [[p, 0] for p in primes_s4]
        initial_states.append(c4)
        
    best_classes = []
    best_L = 0
    best_cost = Fraction(0)
    
    for state in initial_states:
        L = _compute_L_fast(state, 100000)
        cost = sum(Fraction(1, p) for p, r in state)
        if L > best_L and cost <= 1:
            best_L = L
            best_classes = state
            best_cost = cost
    
    used_primes = set(p for p, r in best_classes)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try changing residues
        for idx in range(len(best_classes)):
            if time.time() > deadline: break
            p, r = best_classes[idx]
            for dr in [-1, 1, -2, 2, -3, 3]:
                new_r = (r + dr) % p
                if new_r == r: continue
                
                test_classes = best_classes.copy()
                test_classes[idx] = [p, new_r]
                
                new_L = _compute_L_fast(test_classes, 100000)
                if new_L > best_L:
                    best_L = new_L
                    best_classes = test_classes
                    improved = True
                    break
            
            if improved:
                break
        
        # 2. Try adding a new prime
        if not improved:
            best_new_L = best_L
            best_new_prime = None
            best_new_residue = None
            best_new_cost = best_cost
            
            unused = [p for p in primes_asc if p not in used_primes]
            
            for p in unused[:50]:
                if time.time() > deadline: break
                cost_p = Fraction(1, p)
                if best_cost + cost_p > 1:
                    continue
                
                residues_to_try = {0, 1, 2, (best_L + 1) % p}
                
                for r in residues_to_try:
                    test_classes = best_classes + [[p, r]]
                    new_L = _compute_L_fast(test_classes, 100000)
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_prime = p
                        best_new_residue = r
                        best_new_cost = best_cost + cost_p
            
            if best_new_prime is not None:
                best_classes.append([best_new_prime, best_new_residue])
                best_L = best_new_L
                best_cost = best_new_cost
                used_primes.add(best_new_prime)
                improved = True
        
    final_L = _compute_L_fast(best_classes, 100000)
    if final_L > best_L:
        best_L = final_L
        
    return best_classes
# EVOLVE-BLOCK-END
