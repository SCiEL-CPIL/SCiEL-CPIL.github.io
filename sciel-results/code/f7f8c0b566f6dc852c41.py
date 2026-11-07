import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def sf_kernel(n):
        """Return squarefree kernel of n."""
        if n <= 1:
            return 1
        result = 1
        f = n
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    result *= d
            d += 1 if d == 2 else 2
        if f > 1:
            result *= f
        return result
    
    # Precompute squarefree kernels up to 10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    sf[1] = 1
    for i in range(2, max_val + 1):
        sf[i] = sf_kernel(i)
    
    # Target: squarefree kernel of 210 = 2*3*5*7 = 210
    target = 210
    
    # Check size 1: find n in [211, 10210] with sf[n] == 210
    best_max = max_val  # worst case
    best_list = None
    
    # Size 1
    for n in range(211, max_val + 1):
        if sf[n] == target:
            if n < best_max:
                best_max = n
                best_list = [n]
            break  # first one is smallest
    
    # Build a map from sf value to list of numbers (sorted) with that sf value
    sf_to_nums = defaultdict(list)
    for n in range(211, max_val + 1):
        s = sf[n]
        sf_to_nums[s].append(n)
    for s in sf_to_nums:
        sf_to_nums[s].sort()
    
    # Also keep sf_to_min for quick lookup
    sf_to_min = {}
    for n in range(211, max_val + 1):
        s = sf[n]
        if s not in sf_to_min or n < sf_to_min[s]:
            sf_to_min[s] = n
    
    # For size 2: need sf[a] * sf[b] to have squarefree part 210
    def coprime_factor_pairs(n):
        pairs = []
        for d in range(1, int(math.isqrt(n)) + 1):
            if n % d == 0:
                e = n // d
                if math.gcd(d, e) == 1:
                    pairs.append((d, e))
                    if d != e:
                        pairs.append((e, d))
        return pairs
    
    pairs_210 = coprime_factor_pairs(210)
    
    # For size 2, try all coprime factor pairs and g values
    for x_prime, y_prime in pairs_210:
        if time.time() > deadline:
            break
        max_g = max_val // max(x_prime, y_prime)
        g = 1
        while g <= max_g:
            if math.gcd(g, 210) == 1:
                sa = x_prime * g
                sb = y_prime * g
                if sa in sf_to_min and sb in sf_to_min:
                    a = sf_to_min[sa]
                    b = sf_to_min[sb]
                    if a != b:
                        m = max(a, b)
                        if m < best_max:
                            best_max = m
                            best_list = [a, b]
            g += 1
    
    # BFS over XOR states to find minimal max value
    # State: (current_xor, max_so_far) -> best (min max_so_far)
    # We want to find a set S such that XOR of sf[n] for n in S == target
    # And minimize max(S)
    
    # Use a priority queue approach: explore states in order of max value
    # But that's complex. Instead, use BFS with increasing M.
    
    # Alternative: for each M from 211 to best_max, check if there's a subset
    # of {211, ..., M} with XOR == target.
    
    # To make this efficient, use linear algebra over GF(2).
    # Represent each sf[n] as a bitmask of its prime factors.
    
    # First, collect all primes that appear in any sf[n] for n in [211, 10210]
    all_primes = set()
    for n in range(211, max_val + 1):
        f = sf[n]
        d = 2
        while d * d <= f:
            if f % d == 0:
                all_primes.add(d)
                while f % d == 0:
                    f //= d
            d += 1 if d == 2 else 2
        if f > 1:
            all_primes.add(f)
    
    # Add primes from target
    f = target
    d = 2
    while d * d <= f:
        if f % d == 0:
            all_primes.add(d)
            while f % d == 0:
                f //= d
        d += 1 if d == 2 else 2
    if f > 1:
        all_primes.add(f)
    
    primes = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Build bitmask for each sf value
    def sf_to_mask(ker):
        f = ker
        mask = 0
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1 and d in prime_to_idx:
                    mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if f > 1:
            if f in prime_to_idx:
                mask |= (1 << prime_to_idx[f])
        return mask
    
    target_mask = sf_to_mask(target)
    
    # Precompute masks for all n
    masks = [0] * (max_val + 1)
    for n in range(211, max_val + 1):
        masks[n] = sf_to_mask(sf[n])
    
    # Now, for each M from 211 to best_max, check if there's a subset of
    # {211, ..., M} with XOR == target_mask.
    
    # Use Gaussian elimination to find if target is in the span of {masks[211], ..., masks[M]}
    # And if so, find the subset.
    
    # To be efficient, incrementally add vectors and maintain the basis.
    
    # Basis: list of (mask, rep) where rep is the XOR of original indices
    # rep is stored as a Python int (bitmask of indices)
    
    basis = []  # list of (mask, rep)
    basis_by_pivot = {}  # pivot_bit -> index in basis
    
    def add_to_basis(mask, rep):
        """Add vector (mask, rep) to basis. Return None if it's in the span,
        or the new rep if it extends the basis."""
        m = mask
        r = rep
        for i, (bm, br) in enumerate(basis):
            if m & bm:
                # Find leading bit
                lead = bm.bit_length() - 1
                if m & (1 << lead):
                    m ^= bm
                    r ^= br
                    if m == 0:
                        return None  # in span
        if m == 0:
            return None
        # Find leading bit of m
        lead = m.bit_length() - 1
        # Insert into basis at appropriate position
        # For simplicity, just append and we'll handle pivots later
        basis.append((m, r))
        basis_by_pivot[lead] = len(basis) - 1
        return r
    
    def is_in_span(target_m):
        """Check if target_m is in the span of basis. If so, return the rep."""
        m = target_m
        r = 0
        for i, (bm, br) in enumerate(basis):
            lead = bm.bit_length() - 1
            if m & (1 << lead):
                m ^= bm
                r ^= br
        if m == 0:
            return r
        return None
    
    # Incrementally add vectors from 211 upward, checking if target is in span
    found = False
    for M in range(211, best_max):
        if time.time() > deadline:
            break
        
        # Add masks[M] to basis
        rep = add_to_basis(masks[M], 1 << (M - 211))
        
        # Check if target is now in span
        if rep is not None or True:  # always check
            target_rep = is_in_span(target_mask)
            if target_rep is not None:
                # Extract the set of indices
                selected = []
                for i in range(M - 210):
                    if target_rep & (1 << i):
                        selected.append(211 + i)
                if selected and max(selected) <= M:
                    m = max(selected)
                    if m < best_max:
                        best_max = m
                        best_list = selected
                        found = True
                        break
    
    # If we found a good result via BFS, return it. Otherwise, fall back to previous best.
    if best_list is None:
        best_list = [840]
        best_max = 840
    
    # Ensure all values are Python ints
    result = [int(x) for x in best_list]
    return result
