import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    # Precompute primes up to 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # Target: 9 distinct sums
    # Strategy: A all odd, B all even (or vice versa) to ensure odd sums (primes > 2)
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(2, 129) if i % 2 == 0]
    
    # We will search for A in odds, B in evens
    # To make it faster, we can iterate A and find compatible B's
    
    # Precompute for each odd a, which even b make a+b prime
    # compat[a] = set of b in evens such that a+b is prime
    
    # Actually, let's just iterate combinations of A and check B validity.
    # Number of A's: C(64, 3) ~ 42k
    # For each A, we need to find B.
    
    # Optimization: 
    # For a fixed A, a number b is compatible if a+b is prime for all a in A.
    # Let's precompute prime sums for each a.
    # prime_partners[a] = set of b in [1, 128] such that a+b is prime.
    
    prime_partners = {}
    for a in range(1, 129):
        partners = set()
        for b in range(1, 129):
            if is_prime[a+b]:
                partners.add(b)
        prime_partners[a] = partners
        
    # Now, iterate A from odds
    # For each A, intersection of prime_partners[a] for a in A gives valid B candidates.
    # We need to pick 3 distinct b's from this intersection such that all 9 sums are distinct.
    
    found_solution = None
    
    # To ensure we don't miss, we can also try A in evens, B in odds.
    # But symmetry suggests trying one is enough if we are thorough.
    # Let's iterate A in odds.
    
    # To speed up, we can sort odds or just iterate.
    # We can also break early if time is up.
    
    for A in itertools.combinations(odds, 3):
        if time.time() > deadline:
            break
            
        # Find intersection of prime partners for all a in A
        valid_bs = prime_partners[A[0]]
        for a in A[1:]:
            valid_bs = valid_bs.intersection(prime_partners[a])
            
        if len(valid_bs) < 3:
            continue
            
        # We need to choose 3 distinct b's from valid_bs
        # such that the 9 sums a_i + b_j are all distinct.
        # Note: if b1 != b2, then a_i + b1 != a_i + b2.
        # We need a_i + b_j != a_k + b_l for all distinct pairs.
        # Since b's are distinct, a_i + b_j are distinct for fixed i.
        # We just need to check if sums from different rows collide.
        
        # Iterate through combinations of 3 from valid_bs
        # If valid_bs is large, this might be slow. But usually intersection is small.
        for B in itertools.combinations(sorted(valid_bs), 3):
            sums = set()
            distinct = True
            for a in A:
                for b in B:
                    s = a + b
                    if s in sums:
                        distinct = False
                        break
                    sums.add(s)
                if not distinct:
                    break
            
            if distinct and len(sums) == 9:
                # Check pairwise coprime (primes are coprime, so this is satisfied)
                # Just to be safe, verify coprimality
                coprime = True
                sums_list = list(sums)
                for i in range(len(sums_list)):
                    for j in range(i+1, len(sums_list)):
                        if __import__('math').gcd(sums_list[i], sums_list[j]) != 1:
                            coprime = False
                            break
                    if not coprime:
                        break
                if coprime:
                    found_solution = [list(A), list(B)]
                    break
        
        if found_solution:
            break
    
    if found_solution:
        return found_solution
        
    # Fallback: Random search if structured search fails or times out
    # This handles cases where parity constraint is too tight or we missed something
    import random
    import math
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Evaluate seed
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a+b)
        # Check pairwise coprime
        for i in range(len(sums)):
            for j in range(i+1, len(sums)):
                if math.gcd(list(sums)[i], list(sums)[j]) != 1:
                    return 0
        return len(sums)
        
    current_score = evaluate(best_A, best_B)
    best_score = current_score
    
    while time.time() < deadline - 5.0: # 5s buffer
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if best_score == 9:
                break
        # Local improvement
        # Perturb A or B
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            old_val = A[idx]
            A[idx] = random.randint(1, 128)
            if A.count(A[idx]) > 1: # Ensure distinct
                A[idx] = old_val
            else:
                new_score = evaluate(A, B)
                if new_score > best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_B = B[:]
                else:
                    A[idx] = old_val
        else:
            idx = random.randint(0, 2)
            old_val = B[idx]
            B[idx] = random.randint(1, 128)
            if B.count(B[idx]) > 1:
                B[idx] = old_val
            else:
                new_score = evaluate(A, B)
                if new_score > best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_B = B[:]
                else:
                    B[idx] = old_val
                    
    return [best_A, best_B]
