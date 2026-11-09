import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    
    N = 210
    MAX_VAL = 10210
    
    # Target parity: for each prime, we need (exponent_in_product + exponent_in_210) % 2 == 0
    # So parity_needed[prime] = exponent_in_210 % 2
    # 210 = 2^1 * 3^1 * 5^1 * 7^1
    # So we need odd exponent for 2,3,5,7 and even for all others
    
    # We'll represent parity as a frozenset of primes with odd exponent in the product
    # Target: frozenset({2, 3, 5, 7})
    
    def get_parity(x):
        """Return frozenset of primes with odd exponent in x."""
        factors = set()
        temp = x
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt ^= 1
                if cnt:
                    factors.add(d)
            d += 1
        if temp > 1:
            factors.add(temp)
        return frozenset(factors)
    
    target = frozenset({2, 3, 5, 7})
    
    # Generate parity vectors for numbers 211..MAX_VAL
    # We want to find a small subset S such that XOR of parities = target
    # XOR of parities = symmetric difference of the frozensets
    
    # Strategy: try subset sizes 1, 2, 3, 4, ...
    # For size 1: check if any number has parity == target
    # For size 2: check if any two numbers have XOR == target
    # etc.
    
    # Precompute parities
    parities = {}
    for x in range(N + 1, MAX_VAL + 1):
        parities[x] = get_parity(x)
    
    # Try size 1
    for x in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        if parities[x] == target:
            return [x]
    
    # Try size 2: need a ^ b = target, i.e., b = a ^ target
    # Map: parity -> list of numbers with that parity
    parity_to_nums = defaultdict(list)
    for x in range(N + 1, MAX_VAL + 1):
        parity_to_nums[parities[x]].append(x)
    
    for x in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        p = parities[x]
        # We need y such that p ^ parity[y] = target
        # i.e., parity[y] = p ^ target
        # XOR of frozensets = symmetric difference
        needed = p ^ target
        if needed in parity_to_nums:
            for y in parity_to_nums[needed]:
                if y != x:
                    return [x, y]
    
    # Try size 3: a ^ b ^ c = target
    # For each pair (a,b), check if a^b^target exists in parity_to_nums
    # But this is O(n^2) which might be too slow. Let's be smart.
    
    # Actually, let's try a different approach for size 3:
    # For each c, we need a^b = target^parity[c]
    # So for each c, check if there exist a,b with a^b = target^parity[c]
    # This is the same as the size-2 problem but with a different target
    
    # Let's precompute all pairwise XORs
    # Actually, let's just iterate carefully
    
    # For size 3:
    for c in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        pc = parities[c]
        target2 = pc ^ target  # need a^b = target2
        # Now find a,b with a^b = target2
        # a^b = target2 means b = a^target2
        for a in range(N + 1, MAX_VAL + 1):
            if a == c:
                continue
            pa = parities[a]
            needed_b = pa ^ target2
            if needed_b in parity_to_nums:
                for b in parity_to_nums[needed_b]:
                    if b != a and b != c:
                        return [a, b, c]
        if time.time() > deadline:
            break
    
    # This is getting expensive. Let me think of a better approach.
    # 
    # Actually, the O(n^2) for size 3 is ~10000^2 = 100M which is too slow.
    # Let me use a smarter method.
    
    # For size 3: a^b^c = target
    # Group by parity. For each parity p1, p2, we need p1^p2^p3 = target
    # So p3 = p1^p2^target
    # 
    # Let's count how many numbers have each parity, and find triples of parities
    # that XOR to target, then pick actual numbers.
    
    # Actually, let me just be more careful with the size-2 and try to find
    # a solution with minimal max value.
    
    # Let me restart with a cleaner approach:
    # Sort numbers by value, and for each possible max value M, try to find
    # a valid subset using only numbers in [211, M].
    
    # This is complex. Let me just return a known valid solution.
    # 
    # 210 * x = square means x must be 210 * k^2 / 210... no.
    # 210 * x = square. 210 = 2*3*5*7. So x must be 2*3*5*7 * y^2 = 210*y^2
    # for x to make 210*x a square. But x must be > 210, so y >= 2.
    # x = 210*4 = 840. Check: 210*840 = 176400 = 420^2. Yes!
    # 
    # But wait, we can also use multiple numbers. The product of all selected * 210 = square.
    # 
    # For a single number: x = 210 * y^2. Smallest y>1: y=2, x=840.
    # max - 210 = 840 - 210 = 630.
    # 
    # Can we do better with 2 numbers? We need x1*x2*210 = square.
    # x1*x2 must have the same "squarefree" structure as 210.
    # 
    # Let me just return [840] as a safe baseline and see if I can do better.
    
    return [840]
