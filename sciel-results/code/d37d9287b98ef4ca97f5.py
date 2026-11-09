import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = 60.0 # seconds
    
    # Candidate numbers 2..256
    candidates = list(range(2, P['n'] + 1))
    
    # Initial set: all primes
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    current_set = primes
    best_set = primes
    
    # Greedy addition
    # Try to add composites
    # Check validity of adding x to S
    def can_add(S, x):
        # S is a set of ints
        # x is not in S
        # Check 1: x does not divide b*c for any distinct b,c in S
        # Check 2: a does not divide x*c or b*x for any distinct a,b,c in S (a != b != c != a)
        # Actually, the condition is for ALL distinct a,b,c in S_new.
        # If x is added, we must check:
        # 1. x | b*c for any distinct b,c in S -> FAIL
        # 2. a | x*c for any distinct a,c in S (a!=c, and a,c != x which is true) -> FAIL
        # 3. a | b*x for any distinct a,b in S -> FAIL (same as 2)
        
        S_list = list(S)
        n = len(S_list)
        
        # Check 1: x | b*c
        for i in range(n):
            for j in range(i+1, n):
                b = S_list[i]
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
                    
        # Check 2: a | x*c
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                c = S_list[j]
                if i == j: continue
                if (x * c) % a == 0:
                    return False
                    
        return True

    # Try to add composites to the prime set
    # Sort composites by some heuristic?
    composites = [x for x in candidates if not is_prime(x)]
    
    # We might need to remove some primes to add composites
    # This is a Max Independent Set problem on a conflict graph?
    # Or just local search.
    
    # Since 54 is small, we can try random restarts of greedy.
    
    # Heuristic: Try adding composites one by one.
    # If can't add, try swapping a prime for a composite?
    
    # Let's just try to find a valid set of size 55 using randomization.
    
    best_size = len(best_set)
    
    for _ in range(1000): # Iterations
        if time.time() > deadline - 5:
            break
            
        # Start with primes
        S = set(primes)
        
        # Shuffle composites
        random.shuffle(composites)
        
        added = 0
        for x in composites:
            if time.time() > deadline - 5:
                break
            if x not in S and can_add(S, x):
                S.add(x)
                added += 1
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
