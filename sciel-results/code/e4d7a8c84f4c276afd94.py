import time
import random
import itertools

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for the 90s limit
    
    rng = random.Random(42)
    
    # Candidate 1: Arithmetic Progression 1..20
    # Sumset size for {1..20} is 1+2+...+20+19+...+1? No.
    # Sumset of {1,2,...,n} is {2, 3, ..., 2n}. Size = 2n-1 = 39.
    # Productset of {1,2,...,20} is much larger.
    # Let's estimate productset size.
    # Many products are unique.
    
    # Candidate 2: Geometric Progression
    # {1, 2, 4, ..., 2^19} -> 2^19 = 524288 > 10000.
    # So max power is 2^13 = 8192.
    # {1, 2, 4, ..., 8192} has 14 elements.
    # Sumset of geometric progression:
    # If A = {r^0, ..., r^{k-1}}, sumset is {r^i + r^j}.
    # For r=2, sums are distinct for i != j?
    # 1+2=3, 1+4=5, 2+4=6, 1+8=9, 2+8=10, 4+8=12...
    # Actually, for powers of 2, the sumset size is k(k+1)/2.
    # For k=14, size = 14*15/2 = 105.
    # Productset of geometric progression is also geometric: {r^{i+j}}.
    # Size is k(k+1)/2 = 105.
    # So max size is 105.
    
    # Candidate 3: Small smooth numbers
    # Smooth numbers have many factors, so productset might be smaller?
    # Or larger?
    # If A consists of 2^a 3^b, then A*A consists of 2^{a+c} 3^{b+d}.
    # The set of exponents forms a grid.
    # If we pick 20 distinct pairs (a,b), the productset size is the size of the Minkowski sum of the exponent sets.
    # If we pick a "square" in exponent space, the sumset of exponents is larger.
    # Let's try to pick a set of 20 smooth numbers that minimizes the exponent sumset.
    
    # Let's generate some candidates and evaluate them.
    
    candidates = []
    
    # 1. AP 1..20
    candidates.append(list(range(1, 21)))
    
    # 2. GP 1..8192 (powers of 2)
    gp = [2**i for i in range(14)]
    # Need 20 elements. Add 6 more.
    # If we add more powers of 2, we are limited by 10000.
    # So we can't just extend the GP.
    # Let's try a different base.
    # Base 3: 1, 3, 9, 27, 81, 243, 729, 2187, 6561 (9 elements)
    # Base 4: 1, 4, 16, 64, 256, 1024, 4096 (7 elements)
    # Mixed?
    
    # Let's try a set of small numbers that are "multiplicatively closed" or similar.
    # Actually, the GP of 2s is a very strong candidate for small productset.
    # Sumset size 105, Productset size 105.
    # Can we do better?
    # If we use {1, 2, 3, 4, 6, 8, 9, 12, ...} (smooth numbers)
    # The productset of smooth numbers can be quite large if the exponents spread out.
    
    # Let's evaluate the GP candidate.
    # We need 20 distinct integers.
    # GP of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 nums)
    # Add: 16384? No, >10000.
    # So we must add non-powers of 2.
    # If we add 3, the productset grows.
    # Let's just use the first 20 powers of 2? No, max is 10000.
    # So GP is limited to 14 elements.
    
    # What if we use a GP with ratio close to 1?
    # Not possible with integers.
    
    # What if we use a set of numbers that are all divisors of a highly composite number?
    # E.g., divisors of 2^a 3^b 5^c ...
    # If A is a subset of divisors of N, then A*A is a subset of divisors of N^2.
    # The number of divisors of N^2 can be large.
    
    # Let's try to minimize the sumset first.
    # The minimum sumset size for n integers is 2n-1 (achieved by AP).
    # For n=20, min sumset size is 39.
    # The productset size for AP 1..20 is likely much larger than 39.
    # Let's calculate productset size for 1..20.
    
    def calc_score_fast(A):
        A = sorted(A)
        sumset = set()
        prodset = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    # Evaluate candidates
    best_A = None
    best_score = float('inf')
    
    # Candidate 1: 1..20
    A1 = list(range(1, 21))
    s1 = calc_score_fast(A1)
    if s1 < best_score:
        best_score = s1
        best_A = A1
    
    # Candidate 2: Powers of 2 + some others
    # Try to complete the GP of 2s with small numbers that don't explode the productset.
    # If we add 3, products with 3 will create new values.
    # Let's try {1, 2, 4, ..., 8192} U {3, 6, 12, 24, 48, 96}
    # These are all of the form 2^k or 3*2^k.
    # A = {2^k, 3*2^k} for k=0..13?
    # That's 28 numbers. We need 20.
    # Let's take k=0..9 for 2^k (10 nums) and k=0..9 for 3*2^k (10 nums).
    # Max value: 3*2^9 = 1536 <= 10000.
    # This set is closed under multiplication by 2 and 3? No, just a subset.
    # But the products will be of the form 2^{a+b} or 2^{a+b+1} 3 or 9 2^{a+b}.
    # The exponents of 2 will range from 0 to 18.
    # The exponents of 3 will range from 0 to 2.
    # So the productset is contained in {2^i 3^j | 0<=i<=18, 0<=j<=2}.
    # Size <= 19 * 3 = 57.
    # The sumset?
    # Sums of 2^a and 2^b are distinct (mostly).
    # Sums of 2^a and 3*2^b?
    # 2^a + 3*2^b = 2^b (2^{a-b} + 3).
    # If a > b, this is 2^b (odd + 3).
    # It seems the sumset might be larger.
    # Let's generate this candidate.
    
    A2 = []
    for k in range(10):
        A2.append(2**k)
    for k in range(10):
        A2.append(3 * 2**k)
    # Check distinctness
    if len(set(A2)) == 20:
        s2 = calc_score_fast(A2)
        if s2 < best_score:
            best_score = s2
            best_A = A2
    
    # Candidate 3: Small smooth numbers
    # Generate 20 smallest 2,3-smooth numbers
    smooth = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    A3 = smooth[:20]
    s3 = calc_score_fast(A3)
    if s3 < best_score:
        best_score = s3
        best_A = A3
    
    # Candidate 4: Random search for better sets
    # Start with best_A and do local search
    # Time remaining
    time_left = deadline - time.time()
    if time_left > 1.0:
        # Local search
        current_A = list(best_A)
        current_score = best_score
        
        # Candidate pool: small numbers + smooth numbers
        pool = list(range(1, 101))
        for i in range(1, 14):
            for j in range(1, 14):
                val = (2**i) * (3**j)
                if val <= 10000:
                    pool.append(val)
        pool = list(set(pool))
        rng.shuffle(pool)
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 1.0:
            improved = False
            for i in range(len(current_A)):
                if (time.time() - start_time) > deadline - 1.0:
                    break
                old_val = current_A[i]
                for new_val in pool:
                    if new_val == old_val or new_val in current_A:
                        continue
                    
                    new_A = current_A.copy()
                    new_A[i] = new_val
                    new_score = calc_score_fast(new_A)
                    
                    if new_score < current_score:
                        current_score = new_score
                        current_A = new_A
                        improved = True
                        break
                if (time.time() - start_time) > deadline - 1.0:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_A = current_A
        
    # Final check
    if best_A is None:
        best_A = list(range(1, 21))
        
    return best_A
