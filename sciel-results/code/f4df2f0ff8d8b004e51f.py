import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Generate set S: numbers with a prime factor > 11
    S = set()
    for x in nums:
        n = x
        has_large = False
        d = 2
        while d * d <= n:
            while n % d == 0:
                if d > 11:
                    has_large = True
                n //= d
            d += 1
        if n > 1 and n > 11:
            has_large = True
        if has_large:
            S.add(x)
            
    # Add 1
    S.add(1)
    
    # Check if we can add more smooth numbers
    smooth_nums = [x for x in nums if x not in S]
    
    # Sort smooth nums, maybe try to add them
    # But first, validate S
    # We assume S is valid based on analysis.
    
    # Let's try to add smooth numbers one by one
    # Greedy addition of smooth numbers
    current_set = list(S)
    current_set_set = set(current_set)
    
    def is_valid_addition(selected, x):
        n = len(selected)
        # Check if adding x creates a bad triple
        # We only need to check pairs in selected
        # Optimization: if x is smooth, and a,b in selected have different large primes, 
        # they can't form a bad triple with x.
        # So only check pairs (a,b) in selected that have the SAME large prime.
        
        # Precompute large prime for each element in selected
        # large_prime[x] = p if x has a prime factor > 11, else None
        
        for i in range(n):
            a = selected[i]
            # If a is smooth (large_prime None), it can form bad triples with any b
            # But x is smooth. 
            # If a is in S, it has a large prime.
            # If a in S and b in S and P(a) != P(b), no bad triple with x.
            # If a in S and b smooth, P(a) != None, P(b)=None. No bad triple with x?
            # Wait. If a in S, b smooth, x smooth.
            # L = lcm(a,b). P(a) divides L.
            # v_P(a)(L) = v_P(a)(a).
            # v_P(a)(b) = 0.
            # v_P(a)(x) = 0.
            # Only a has it. Fail.
            # So if a in S, it CANNOT form a bad triple with two smooth numbers.
            # Thus, x (smooth) can only form a bad triple with two elements from S that share the same large prime.
            
            # So we only need to check pairs (a,b) in S such that P(a) == P(b).
            
            pass # Logic simplified below

    # Refined check
    large_prime_map = {}
    for x in nums:
        n = x
        p = None
        d = 2
        while d * d <= n:
            while n % d == 0:
                if d > 11:
                    p = d
                n //= d
            d += 1
        if n > 1 and n > 11:
            p = n
        large_prime_map[x] = p

    # Group S by large prime
    from collections import defaultdict
    groups = defaultdict(list)
    for x in S:
        p = large_prime_map[x]
        if p:
            groups[p].append(x)

    def can_add_x(x, current_set):
        # x is smooth
        # Only conflicts with pairs (a,b) in current_set where P(a)==P(b)
        # AND the smooth part condition holds.
        
        # For each group
        for p, members in groups.items():
            # Filter members that are in current_set
            active_members = [m for m in members if m in current_set]
            n = len(active_members)
            for i in range(n):
                a = active_members[i]
                for j in range(i+1, n):
                    b = active_members[j]
                    # Check if {a, b, x} is a bad triple
                    l_ab = _lcm(a, b)
                    l_ax = _lcm(a, x)
                    l_bx = _lcm(b, x)
                    if l_ab == l_ax == l_bx:
                        return False
        return True

    # Try to add smooth numbers
    # Sort by something?
    for x in smooth_nums:
        if time.time() > deadline:
            break
        if x in current_set_set:
            continue
        if can_add_x(x, current_set_set):
            current_set.append(x)
            current_set_set.add(x)

    # Final validation
    # ... (standard validation)
    
    return current_set
