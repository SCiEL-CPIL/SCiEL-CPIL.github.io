import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def is_valid_addition(S, x):
        """
        Check if x can be added to S.
        S is a list of ints.
        Conditions:
        1. For all a in S: a does not divide x*c for any c in S, c != a.
           Equivalent to: For all a in S, let k = a // gcd(a, x).
           No c in S (c != a) is a multiple of k.
        2. For all a in S: a does not divide b*x for any b in S, b != a.
           Equivalent to: For all a in S, let k = a // gcd(a, x).
           No b in S (b != a) is a multiple of k.
           (Same check as 1, just different variable name)
        3. For all distinct b, c in S: x does not divide b*c.
        """
        # Optimization: If x is in S, return False (distinctness)
        # We assume x not in S.
        
        # Check 1 & 2: a | x*c and a | b*x
        # For each a in S, k_a = a // gcd(a, x)
        # We need to ensure no other element in S is a multiple of k_a.
        
        # Check 3: x | b*c
        # Iterate pairs in S.
        
        S_set = set(S)
        
        # Check 1 & 2
        for a in S:
            g = gcd(a, x)
            k = a // g
            if k == 1:
                # a divides x.
                # Then a divides x*c for ALL c.
                # So if there is any c in S, c != a, it's a violation.
                if len(S) > 1:
                    return False
                # If len(S) == 1, S = [a]. No other c or b.
                # So this check passes for 1 & 2.
                continue
            
            # Check if any c in S (c != a) is a multiple of k
            # We can iterate S.
            for c in S:
                if c == a:
                    continue
                if c % k == 0:
                    return False
        
        # Check 3: x | b*c
        # If S is small, O(N^2) is fine.
        # S size <= 255.
        n = len(S)
        for i in range(n):
            b = S[i]
            for j in range(i + 1, n):
                c = S[j]
                if (b * c) % x == 0:
                    return False
        
        return True

    def greedy(order):
        S = []
        # Pre-sort order? No, order is given.
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.append(x)
        return S

    best_set = []
    
    # Strategy 1: Descending order
    order1 = list(range(256, 1, -1))
    S1 = greedy(order1)
    if len(S1) > len(best_set):
        best_set = S1
        
    # Strategy 2: Primes first, then composites descending
    primes = [a for a in range(2, 257) if all(a % d for d in range(2, int(a**0.5) + 1))]
    composites = [a for a in range(2, 257) if a not in primes]
    order2 = primes + sorted(composites, reverse=True)
    S2 = greedy(order2)
    if len(S2) > len(best_set):
        best_set = S2
        
    # Strategy 3: Random shuffles
    random.seed(42)
    all_nums = list(range(2, 257))
    for i in range(50): # Try 50 random orders
        if time.time() > deadline:
            break
        random.shuffle(all_nums)
        Sr = greedy(all_nums)
        if len(Sr) > len(best_set):
            best_set = Sr
            
    # Strategy 4: Sort by number of divisors (ascending) - primes have 2, highly composite have many
    # This might prefer primes?
    order4 = sorted(all_nums, key=lambda x: len(get_divisors(x)))
    S4 = greedy(order4)
    if len(S4) > len(best_set):
        best_set = S4
        
    # Final check
    # Convert to list of ints
    return [int(x) for x in best_set]
