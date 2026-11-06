"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    from itertools import combinations
    
    # Build a large valid set using structured approach
    # Take powers of 2, and numbers of form p * 2^k for distinct odd primes p
    # with careful selection to avoid equal-GCD triples
    
    # Start with powers of 2: 1, 2, 4, 8, 16, 32, 64
    base = [1, 2, 4, 8, 16, 32, 64]
    
    # Add multiples: for each odd prime p, add p, 2p, 4p, ... up to 128
    # But we need to check the constraint
    # 
    # Key insight: gcd(2^a * m1, 2^b * m2) = 2^min(a,b) * gcd(m1, m2)
    # If m1, m2 are distinct odd primes, gcd(m1,m2)=1
    # So gcd(2^a * p, 2^b * q) = 2^min(a,b) for p≠q distinct odd primes
    #
    # For three numbers 2^a*p, 2^b*q, 2^c*r with distinct odd primes p,q,r:
    # gcds are 2^min(a,b), 2^min(a,c), 2^min(b,c)
    # These are equal iff min(a,b)=min(a,c)=min(b,c), which requires a=b=c
    # But if a,b,c are distinct, at least two mins differ.
    # Actually if a<b<c: min(a,b)=a, min(a,c)=a, min(b,c)=b. So two are a, one is b. Not all equal.
    #
    # For three numbers 2^a*p, 2^b*p, 2^c*p (same odd prime p):
    # gcds: 2^min(a,b)*p, 2^min(a,c)*p, 2^min(b,c)*p
    # If a<b<c: p*2^a, p*2^a, p*2^b. Not all equal.
    #
    # For mix: 2^a*p, 2^b*q, 2^c*p where p≠q:
    # gcd(2^a*p, 2^b*q) = 2^min(a,b)
    # gcd(2^a*p, 2^c*p) = 2^min(a,c)*p
    # gcd(2^b*q, 2^c*p) = 2^min(b,c)
    # These are equal only in special cases.
    
    # Strategy: take all numbers of form 2^k * p where p is an odd prime or p=1
    # i.e., all numbers whose odd part is 1 or a single prime
    # This includes: powers of 2, and p*2^k for each odd prime p
    
    # Let's build the candidate set: all n in [1,128] where the odd part is 1 or prime
    candidates = []
    for n in range(1, 129):
        m = n
        while m % 2 == 0:
            m //= 2
        # m is the odd part
        # Check if m is 1 or prime
        if m == 1:
            candidates.append(n)
        else:
            # Check if m is prime
            is_prime = True
            if m < 2:
                is_prime = False
            else:
                for d in range(3, int(m**0.5) + 1, 2):
                    if m % d == 0:
                        is_prime = False
                        break
                if m == 1:
                    is_prime = False
            if is_prime:
                candidates.append(n)
    
    # Now greedily select from candidates to maximize size while maintaining constraint
    # Sort by some heuristic
    selected = []
    
    def is_valid(new_elem, current_set):
        """Check if adding new_elem to current_set maintains the no-equal-GCD-triple property."""
        for a, b in combinations(current_set, 2):
            g_ab = gcd(a, b)
            g_an = gcd(a, new_elem)
            g_bn = gcd(b, new_elem)
            if g_ab == g_an == g_bn:
                return False
        return True
    
    # Greedy: try adding candidates in order that maximizes final size
    # Try multiple orderings and pick the best
    import random
    
    best_selected = []
    
    for attempt in range(20):
        cand_copy = candidates[:]
        if attempt == 0:
            # Natural order
            pass
        elif attempt == 1:
            cand_copy.sort(reverse=True)
        elif attempt == 2:
            # Sort by number of prime factors (fewer first)
            def num_prime_factors(x):
                count = 0
                d = 2
                while d * d <= x:
                    while x % d == 0:
                        count += 1
                        x //= d
                    d += 1
                if x > 1:
                    count += 1
                return count
            cand_copy.sort(key=num_prime_factors)
        else:
            random.shuffle(cand_copy)
        
        sel = []
        for c in cand_copy:
            if is_valid(c, sel):
                sel.append(c)
        
        if len(sel) > len(best_selected):
            best_selected = sel
    
    # Also try starting with different bases
    # Try adding all powers of 2 first, then others
    for attempt in range(10):
        sel = [1, 2, 4, 8, 16, 32, 64]
        remaining = [c for c in candidates if c not in sel]
        if attempt == 0:
            pass
        elif attempt == 1:
            remaining.sort(reverse=True)
        else:
            random.shuffle(remaining)
        
        for c in remaining:
            if is_valid(c, sel):
                sel.append(c)
        
        if len(sel) > len(best_selected):
            best_selected = sel
    
    return best_selected
# EVOLVE-BLOCK-END
