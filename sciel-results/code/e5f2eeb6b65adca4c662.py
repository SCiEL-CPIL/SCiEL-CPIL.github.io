import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Candidate set: Integers in (128, 256]
    # This set is primitive (no element divides another), but may violate a | bc
    candidates = list(range(129, 257))
    
    # Also consider primes as a fallback/baseline
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [p for p in range(2, 257) if is_prime(p)]
    best_set = set(primes)
    
    # Greedy pruning of the large set
    S = set(candidates)
    
    def find_violating_triple(S):
        # Returns a triple (a, b, c) in S such that a | b*c
        # Optimization: Check pairs (b, c) and see if any a in S divides b*c
        # To make it faster, iterate a, then b, then check if c exists?
        # Or just random sampling?
        # For 128 elements, 128^3 is ~2M ops. Might be slow in Python.
        # Let's try structured search.
        
        # Sort S
        sorted_S = sorted(S)
        n = len(sorted_S)
        
        # Check for a in S:
        for i in range(n):
            a = sorted_S[i]
            # If a is prime, it can only divide b*c if it divides b or c.
            # Since S is primitive, a does not divide b or c.
            # So primes in S never cause violations of type a | b*c?
            # Wait. If a is prime, a | b*c implies a|b or a|c.
            # Since S is primitive (no x|y for x!=y), a cannot divide b or c.
            # So primes in S are SAFE from being the divisor 'a'.
            
            # So we only need to check composite 'a'.
            if is_prime(a):
                continue
                
            # Check b, c in S
            for j in range(n):
                if j == i: continue
                b = sorted_S[j]
                for k in range(n):
                    if k == i or k == j: continue
                    c = sorted_S[k]
                    if (b * c) % a == 0:
                        return (a, b, c)
        return None

    # This triple loop is O(N^3). N=128. 128^3 = 2,097,152.
    # In Python, this might take ~1-2 seconds per check.
    # We might need many iterations.
    
    # Faster check:
    # For each a in S (composite), for each b in S \ {a},
    # we need to know if there exists c in S \ {a, b} such that a | b*c.
    # This is equivalent to: Let g = gcd(a, b). Let a' = a/g.
    # We need c such that a' | c * (b/g).
    # Since gcd(a', b/g) = 1, we need a' | c.
    # So we need to check if there exists c in S \ {a, b} such that c is a multiple of a'.
    
    def find_violating_triple_fast(S):
        sorted_S = sorted(S)
        n = len(sorted_S)
        # Precompute multiples?
        # Just loop a, b.
        for i in range(n):
            a = sorted_S[i]
            if is_prime(a):
                continue
            for j in range(n):
                if j == i: continue
                b = sorted_S[j]
                g = __import__('math').gcd(a, b)
                a_reduced = a // g
                # We need c in S \ {a, b} such that c % a_reduced == 0
                # Since S is primitive, c cannot be a multiple of a (if a_reduced == 1, c can be anything? No, if a_reduced==1, a|b, impossible).
                # So a_reduced > 1.
                # Check if any c in S is divisible by a_reduced.
                # Note: c cannot be a or b.
                # If a_reduced == a, then c must be multiple of a. Since S primitive, no such c (except a itself, excluded).
                # If a_reduced < a, c could be a multiple of a_reduced.
                
                found_c = None
                for k in range(n):
                    if k == i or k == j: continue
                    c = sorted_S[k]
                    if c % a_reduced == 0:
                        found_c = c
                        break
                if found_c:
                    return (a, b, found_c)
        return None

    # Run pruning
    while time.time() < deadline:
        triple = find_violating_triple_fast(S)
        if triple is None:
            break
        
        a, b, c = triple
        # Heuristic: Remove the element with the most divisors in S?
        # Or the largest?
        # Let's remove the one with the highest value, as it's more likely to be composite and "heavy".
        # Or remove the one that is most composite.
        
        # Calculate "compositeness" or just remove max
        # Removing max is a good heuristic for primitive sets.
        to_remove = max(a, b, c)
        S.remove(to_remove)
        
        if len(S) > len(best_set):
            best_set = set(S)
            
    # Ensure best_set is valid (it should be, but just in case)
    # Also compare with primes
    if len(primes) > len(best_set):
        best_set = set(primes)
        
    # Final validation
    def is_valid(S):
        L = list(S)
        for i in range(len(L)):
            for j in range(len(L)):
                if i == j: continue
                for k in range(len(L)):
                    if i == k or j == k: continue
                    if (L[j] * L[k]) % L[i] == 0:
                        return False
        return True
        
    if not is_valid(best_set):
        # Fallback to primes
        best_set = set(primes)
        
    return list(best_set)
