import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i in range(2, limit+1) if is_prime[i]]

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify_solution(selected):
    """Verify that product of selected * 210 is a perfect square."""
    prod = 210
    for x in selected:
        prod *= x
    return _is_perfect_square(prod)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Generate primes up to 10210
    max_val = 10210
    primes = _sieve(max_val)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    num_bits = len(primes)
    
    # Target: parity of 210
    target_mask = 0
    n = 210
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp % 2 == 1 and d in prime_to_bit:
                target_mask |= (1 << prime_to_bit[d])
        d += 1
    if n > 1 and n in prime_to_bit:
        target_mask |= (1 << prime_to_bit[n])
        
    # Basis for Gaussian Elimination
    # rows: list of (mask, frozenset_of_indices)
    rows = []
    pivot_cols = {} # bit_index -> row_index
    
    # We iterate M from 211 to 10210
    # We want the first M where target is in span, and solution size <= 100
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        # Compute mask for M
        m = 0
        n = M
        d = 2
        while d * d <= n:
            if n % d == 0:
                exp = 0
                while n % d == 0:
                    n //= d
                    exp += 1
                if exp % 2 == 1 and d in prime_to_bit:
                    m |= (1 << prime_to_bit[d])
            d += 1
        if n > 1 and n in prime_to_bit:
            m |= (1 << prime_to_bit[n])
            
        # Add M to basis
        # m_temp is the reduced mask
        # set_temp is the set of original indices
        m_temp = m
        set_temp = frozenset([M])
        
        while m_temp != 0:
            hsb = m_temp.bit_length() - 1
            if hsb in pivot_cols:
                row = rows[pivot_cols[hsb]]
                m_temp ^= row[0]
                set_temp = set_temp ^ row[1]
            else:
                break
        
        if m_temp != 0:
            hsb = m_temp.bit_length() - 1
            pivot_cols[hsb] = len(rows)
            rows.append((m_temp, set_temp))
            
        # Check if target_mask is in span
        # Reduce target_mask using current basis
        m_check = target_mask
        set_check = frozenset()
        representable = True
        
        while m_check != 0:
            hsb = m_check.bit_length() - 1
            if hsb in pivot_cols:
                row = rows[pivot_cols[hsb]]
                m_check ^= row[0]
                set_check = set_check ^ row[1]
            else:
                representable = False
                break
        
        if representable:
            # We have a solution
            selected = sorted(list(set_check))
            
            # Check constraints
            if 1 <= len(selected) <= 100:
                # Verify just in case
                if _verify_solution(selected):
                    # This is the minimal M with a valid solution
                    # Return it
                    return [int(x) for x in selected]
            # If size > 100, we continue to next M
            # Note: This might miss a valid solution for this M if the 
            # minimal weight solution is <= 100 but this one is not.
            # However, for small M, this is unlikely.
            
    # Fallback
    return [224, 240]
