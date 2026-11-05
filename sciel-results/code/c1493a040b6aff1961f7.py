import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (limit is 90s)

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    # Precompute primes up to 256 (max sum is 128+128=256)
    primes = sieve(256)
    prime_set = set(primes)
    odd_primes = [p for p in primes if p > 2]
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Strategy: 
    # If all 9 sums are distinct primes, they must be pairwise coprime (since primes > 1 are coprime to each other).
    # For A+B to consist of odd primes, one set must be all odd and the other all even.
    # Let A = {a1, a2, a3} be odd, B = {b1, b2, b3} be even.
    # The sums are a_i + b_j.
    # We want these 9 sums to be distinct primes.
    
    # Alternative strategy:
    # Search for A, B such that the distinct sums are pairwise coprime.
    # We can relax the "all prime" constraint.
    # But "all prime" is a strong sufficient condition.
    # Let's first try to find a solution with 9 distinct prime sums.
    # If not found, try to find a solution with 9 distinct pairwise coprime sums (not necessarily prime).
    
    # Optimization for prime sum search:
    # Fix B. For each B, check if there exists A such that A+B are all primes.
    # Actually, iterating over triples of primes is efficient.
    
    # Let's refine the search for 9 distinct primes.
    # We iterate over possible differences d2, d3 for B (even numbers).
    # B = {b, b+d2, b+d3} where b, b+d2, b+d3 are even.
    # d2, d3 must be even.
    # Let d2 = 2*k2, d3 = 2*k3.
    # Sums are a1+b, a1+b+d2, a1+b+d3, etc.
    # Let S1 = a1+b, S2 = a2+b, S3 = a3+b.
    # Then the sums are:
    # S1, S1+d2, S1+d3
    # S2, S2+d2, S2+d3
    # S3, S3+d2, S3+d3
    # All 9 must be primes.
    
    # We can iterate over d2, d3 (even) and b (even).
    # For a fixed b, d2, d3, we need to find a1, a2, a3 (odd) such that:
    # a1+b, a1+b+d2, a1+b+d3 are prime
    # a2+b, a2+b+d2, a2+b+d3 are prime
    # a3+b, a3+b+d2, a3+b+d3 are prime
    
    # This is equivalent to finding 3 distinct odd numbers a such that a, a+d2, a+d3 are prime (after shifting by b).
    # Let x = a+b. Since b is even and a is odd, x is odd.
    # We need x, x+d2, x+d3 to be prime.
    # So we need to find 3 distinct odd numbers x1, x2, x3 such that:
    # x1, x1+d2, x1+d3 are prime
    # x2, x2+d2, x2+d3 are prime
    # x3, x3+d2, x3+d3 are prime
    
    # And a_i = x_i - b must be in [1, 128].
    # b is in [1, 128] and even.
    # x_i = a_i + b. Since a_i in [1, 128], x_i in [1+b, 128+b].
    # Max x_i is 256.
    
    # So the algorithm is:
    # 1. Iterate over even d2, d3 with 0 < d2 < d3 < 256 (actually d3 < 256-b_min, but let's keep it simple).
    #    Note: d2, d3 are differences in B. B elements are in [1,128].
    #    b_min >= 1, b_max <= 128.
    #    b+d3 <= 128 => d3 <= 127-b. Since b>=1, d3 <= 126.
    #    So d2, d3 are even numbers in [2, 126].
    
    # 2. For each (d2, d3), find all odd x in [1, 256] such that x, x+d2, x+d3 are prime.
    #    Let this set be X.
    # 3. If |X| >= 3, pick 3 distinct x1, x2, x3 from X.
    # 4. For each triple, find b such that a_i = x_i - b are in [1, 128] and distinct.
    #    a_i = x_i - b.
    #    1 <= x_i - b <= 128 => x_i - 128 <= b <= x_i - 1.
    #    So b must be in [max(x_i - 128), min(x_i - 1)] for all i.
    #    Also b must be even and in [1, 128].
    #    If such a b exists, we have a valid solution.
    
    # This is much faster because we precompute X for each (d2, d3).
    
    possible_diffs = [2*d for d in range(1, 64)] # 2 to 126
    
    # Precompute valid x for each pair of diffs? No, just loop.
    
    # To speed up, we can iterate over d2, d3 and collect valid x.
    
    for d2 in possible_diffs:
        if time.time() > deadline:
            break
        for d3 in possible_diffs:
            if d3 <= d2:
                continue
            if time.time() > deadline:
                break
            
            # Find all x such that x, x+d2, x+d3 are prime
            valid_x = []
            for x in range(1, 257, 2): # x is odd
                if x + d3 > 256:
                    break
                if x in prime_set and (x + d2) in prime_set and (x + d3) in prime_set:
                    valid_x.append(x)
            
            if len(valid_x) < 3:
                continue
            
            # Try combinations of 3 x's
            # Limit to first few if too many?
            # valid_x is likely small.
            triples_to_try = list(itertools.combinations(valid_x, 3))
            
            for x1, x2, x3 in triples_to_try:
                # Determine range for b
                # b <= min(x1, x2, x3) - 1
                # b >= max(x1, x2, x3) - 128
                # b >= 1
                # b <= 128
                # b is even
                
                upper_b = min(x1, x2, x3) - 1
                lower_b = max(x1, x2, x3) - 128
                lower_b = max(1, lower_b)
                upper_b = min(128, upper_b)
                
                if lower_b > upper_b:
                    continue
                
                # Find an even b in [lower_b, upper_b]
                start_even = lower_b if lower_b % 2 == 0 else lower_b + 1
                if start_even > upper_b:
                    continue
                
                b = start_even # Just pick the first one. Any b works for this triple.
                
                a1 = x1 - b
                a2 = x2 - b
                a3 = x3 - b
                
                A = [a1, a2, a3]
                B = [b, b + d2, b + d3]
                
                # Verify distinctness and bounds (should be guaranteed by construction)
                if len(set(A)) != 3 or len(set(B)) != 3:
                    continue
                if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
                    continue
                
                # Verify sums are distinct primes
                sums = set()
                valid = True
                for a in A:
                    for bb in B:
                        s = a + bb
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                        if s not in prime_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(sums) == 9:
                    best_score = 9
                    best_A = A
                    best_B = B
                    return [best_A, best_B]
            
            # If we didn't return, we didn't find a solution with 9 distinct primes for this d2, d3.
            # We can continue searching.
            # But maybe we should check for non-prime pairwise coprime sums too?
            # For now, let's stick to primes. If we exhaust search and don't find 9, we might need to relax.
            # But 9 distinct primes is a strong target.

    # If we didn't find 9 distinct primes, try a broader search?
    # Or just return the best found so far?
    # The current code returns best_A, best_B which might be the seed if no improvement.
    # But the seed has score 5 (or 1.8 normalized?).
    # The task says "Maximize the cardinality of A+B".
    # If we don't find 9, we might find 8 or 7.
    
    # Let's add a fallback search for pairwise coprime sums (not necessarily prime) if time permits.
    # But primes are the easiest way to ensure pairwise coprime.
    
    # If we reach here, we didn't find a 9-prime solution.
    # This is unlikely given the density of primes.
    # Let's double check the logic.
    # Is it possible that no such triple exists?
    # For d2=2, d3=4: x, x+2, x+4 primes. (Prime triplets like 3,5,7; 5,7,9(no); 13,15(no); 17,19,21(no); 101,103,105(no)).
    # 3,5,7 is a triplet. x=3.
    # Next triplet?
    # It is known that there are infinitely many prime triplets? Not proven, but many exist.
    # We need 3 distinct x's.
    # If d2=2, d3=6: x, x+2, x+6.
    # 3,5,9(no). 5,7,11 (yes). 11,13,17 (yes). 17,19,23 (yes).
    # So for d2=2, d3=6, we have x=5, 11, 17.
    # Let's trace this case in the code.
    # d2=2, d3=6.
    # valid_x will contain 5, 11, 17, ...
    # Triple (5, 11, 17).
    # upper_b = min(5,11,17)-1 = 4.
    # lower_b = max(5,11,17)-128 = 17-128 = -111 -> 1.
    # b in [1, 4], even. b=2 or b=4.
    # Let b=2.
    # A = [5-2, 11-2, 17-2] = [3, 9, 15].
    # B = [2, 4, 8].
    # Sums:
    # 3+2=5, 3+4=7, 3+8=11
    # 9+2=11 (Duplicate! 11 is already in sums)
    # So distinct sums check fails.
    # We need x1, x2, x3 such that the sets {x_i, x_i+d2, x_i+d3} are disjoint.
    # In the example above:
    # Set 1: {5, 7, 11}
    # Set 2: {11, 13, 17} -> Intersection {11}
    # Set 3: {17, 19, 23} -> Intersection {17} with Set 2.
    # So we need disjoint prime triplets.
    
    # Let's modify the inner loop to check for disjointness.
    
    # Restart the search with disjointness check.
    # We already have the structure. Let's just re-run the logic with the check.
    # Actually, the previous loop checked `if s in sums`. So it would have rejected it.
    # It would continue to the next triple.
    
    # So the logic is correct. It just needs to find a valid triple.
    
    # If we still don't find one, we return the best found (which might be seed).
    # But the seed is 1.8. We expect to find 9.
    
    return [best_A, best_B]
