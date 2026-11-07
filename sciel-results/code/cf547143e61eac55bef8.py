import time
import random
from itertools import combinations

DEADLINE = 50.0

def _valid_add(C, a):
    # C is a list of current members
    # check for all distinct b,c in C: a does not divide b*c
    # and b does not divide a*c
    # and c does not divide a*b
    n = len(C)
    for i in range(n):
        b = C[i]
        # a, b distinct by construction
        if (b * a) % a == 0: # a | a*b is always true, but b,c must be distinct from a?
            # The condition is: for THREE DISTINCT members a,b,c.
            # If we add a, we form triples (a, b, c) for b,c in C.
            # We need a | b*c to be false.
            # We also need b | a*c to be false.
            # We also need c | a*b to be false.
            pass
    # Correct check:
    for i in range(n):
        b = C[i]
        for j in range(i+1, n):
            c = C[j]
            # Triple (a, b, c)
            if (b * c) % a == 0: return False
            if (a * c) % b == 0: return False
            if (a * b) % c == 0: return False
    return True

def construct():
    t0 = time.time()
    universe = list(range(2, 257))
    
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0: return False
        i = 3
        while i * i <= x:
            if x % i == 0: return False
            i += 2
        return True
        
    primes = [x for x in universe if is_prime(x)]
    
    best_size = 0
    best_set = set()
    
    # Try different P sets
    # P is a set of primes we INCLUDE in the final set S.
    # The rest of S is a valid subset of numbers coprime to product of P.
    
    # Heuristic: Try P = first k primes? Or P = empty?
    # Let's try P = empty, {2}, {2,3}, {2,3,5}, etc.
    # Also try random P subsets.
    
    def solve_for_P(P):
        # P is a set of primes
        prod_P = 1
        for p in P:
            prod_P *= p
            
        # Universe for C
        U = [x for x in universe if x not in P and x % prod_P != 0] 
        # Wait, x % prod_P != 0 is not correct for gcd.
        # We need gcd(x, prod_P) == 1.
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
            
        U = [x for x in universe if x not in P and gcd(x, prod_P) == 1]
        
        # Greedy fill C from U
        # Sort U to get good results?
        # Larger numbers are less likely to divide products of smaller ones?
        # Or smaller numbers are more likely to divide?
        # If a is small, it's easier to divide b*c.
        # So maybe exclude small numbers?
        # Let's try sorting descending.
        C = []
        for x in sorted(U, reverse=True):
            if _valid_add(C, x):
                C.append(x)
                
        S = set(P) | set(C)
        return S
        
    # Try specific P
    P_candidates = [
        set(),
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 5},
        {3, 5},
    ]
    
    for P in P_candidates:
        if time.time() - t0 > DEADLINE: break
        S = solve_for_P(P)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Try random P subsets
    random.seed(42)
    for _ in range(50):
        if time.time() - t0 > DEADLINE: break
        P = set(random.sample(primes, random.randint(0, 5)))
        S = solve_for_P(P)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    result = sorted(best_set)
    return [int(x) for x in result]
