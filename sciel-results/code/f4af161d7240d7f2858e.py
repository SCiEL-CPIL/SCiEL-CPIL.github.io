import math
import time
from collections import defaultdict

def is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    # 210 = 2 * 3 * 5 * 7
    # We need subset of {211, ..., M} such that 210 * prod(subset) is perfect square
    # i.e., the parity vector of prod(subset) must equal parity vector of 210
    # (odd exponents for primes 2,3,5,7; even for all others)
    
    # Factor numbers 211..10210 and compute their square-free parity vectors
    # We'll use a list of primes up to sqrt(10210) ~ 101
    
    max_val = 10210
    # Generate primes up to max_val (actually up to max_val since numbers can be prime)
    # For parity, we just need to factor each number
    
    def factor_parity(n):
        """Return dict: prime -> exponent mod 2 for n"""
        parity = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                parity[d] = parity.get(d, 0) ^ 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            parity[n] = 1
        return parity
    
    # Precompute parity vectors for 211 to some limit
    # Use tuples as dict keys for efficiency
    # We'll represent parity as a frozenset of primes with odd exponent
    
    def get_parity_set(n):
        """Return frozenset of primes with odd exponent in n"""
        ps = set()
        d = 2
        while d * d <= n:
            cnt = 0
            while n % d == 0:
                cnt ^= 1
                n //= d
            if cnt:
                ps.add(d)
            d += 1 if d == 2 else 2
        if n > 1:
            ps.add(n)
        return frozenset(ps)
    
    # Target parity set (from 210): {2, 3, 5, 7}
    target = frozenset([2, 3, 5, 7])
    
    # For each M from 211 upward, maintain a linear basis over GF(2)
    # of the parity vectors of {211, ..., M}
    # Check if target is in the span
    
    # Represent parity vectors as integers (bit vectors) for fast XOR
    # Assign each prime a bit position
    
    # First, collect all primes that appear in 211..10210
    # Actually, let's just use the frozenset representation with a smarter approach
    
    # Better: use integer bit vectors
    # Collect all primes up to 10210
    def sieve_primes(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve_primes(max_val)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Convert parity set to integer bit vector
    def parity_to_int(pset):
        val = 0
        for p in pset:
            idx = prime_to_idx[p]
            val |= (1 << idx)
        return val
    
    # Precompute parity integers for 211..max_val
    parity_ints = [0] * (max_val + 1)
    for n in range(211, max_val + 1):
        parity_ints[n] = parity_to_int(get_parity_set(n))
    
    target_int = parity_to_int(target)
    
    # Gaussian elimination over GF(2)
    # basis: list of (pivot_bit, vector_int, coefficient_bitset)
    # We track which original numbers combine to give each basis vector
    
    best_solution = [224, 240]  # fallback
    best_M = 240
    
    basis = []  # list of [pivot, vec, coeff_mask]
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        
        vec = parity_ints[M]
        
        # Try to reduce vec by existing basis
        reduced_vec = vec
        reduced_coeff = (1 << (M - 211))  # bit (M-211) is set, representing this element
        temp_vec = vec
        temp_coeff = reduced_coeff
        
        for pv, bvec, bcoeff in basis:
            if temp_vec & (1 << pv):
                temp_vec ^= bvec
                temp_coeff ^= bcoeff
        
        if temp_vec == 0:
            # target is in span? No - this means vec is in span, not that target is in span
            # We need to check if target_int is in span separately
            pass
        else:
            # Add to basis
            pivot = temp_vec.bit_length() - 1
            basis.append([pivot, temp_vec, temp_coeff])
            # Sort by pivot descending
            basis.sort(key=lambda x: x[0], reverse=True)
            # Reduce other basis vectors by this one
            for i in range(len(basis)):
                if i == len(basis) - 1:
                    continue
                if basis[i][1] & (1 << pivot):
                    basis[i][1] ^= temp_vec
                    basis[i][2] ^= temp_coeff
        
        # Check if target_int is in the span
        # Reduce target_int by basis
        check_vec = target_int
        for pv, bvec, bcoeff in basis:
            if check_vec & (1 << pv):
                check_vec ^= bvec
            else:
                # Since basis is sorted by pivot descending and we're checking each,
                # if the pivot bit isn't set, skip
                pass
        
        if check_vec == 0:
            # target is in span! But we need the actual combination.
            # Let's redo with coefficient tracking
            check_vec = target_int
            check_coeff = 0
            for pv, bvec, bcoeff in basis:
                if check_vec & (1 << pv):
                    check_vec ^= bvec
                    check_coeff ^= bcoeff
            
            if check_vec == 0:
                # Extract which numbers are selected
                selected = []
                for bit in range(M - 211 + 1):
                    if check_coeff & (1 << bit):
                        selected.append(211 + bit)
                
                if len(selected) >= 1:
                    # Verify
                    prod = 210
                    for s in selected:
                        prod *= s
                    if is_perfect_square(prod):
                        best_solution = selected
                        best_M = M
                        return [int(x) for x in selected]
        
        if time.time() > deadline:
            break
    
    return [int(x) for x in best_solution]
