import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # We want to select a subset of primes with sum of reciprocals <= 1
    # and maximize coverage.
    # Strategy: Try different subsets.
    # Subset 1: Seed (13..1000)
    # Subset 2: 5..97
    # Subset 3: 3..53 (sum check)
    
    best_L = 0
    best_classes = []
    
    # Define candidate sets of primes
    # We will generate sets by taking the first K primes, then maybe swapping small ones for larger ones to reduce cost.
    
    # Let's try to find the largest prefix covered by a greedy assignment for a given set of primes.
    def evaluate_primes(prime_set):
        if time.time() > deadline:
            return 0, []
            
        # Check sum
        total_cost = sum(1/p for p in prime_set)
        if total_cost > 1.0:
            return 0, []
            
        # Greedy assignment of residues
        # We simulate coverage of 1..MAX_L
        MAX_L = 10000
        covered = [False] * (MAX_L + 1)
        classes = []
        
        # Sort primes by size? Or by 1/p?
        # Smaller primes cover more, so assign them first to maximize impact.
        for p in sorted(prime_set):
            best_r = 0
            best_count = -1
            for r in range(p):
                count = 0
                # Check how many new integers are covered
                # Optimization: only check up to MAX_L
                for n in range(1, MAX_L + 1):
                    if not covered[n] and (n - r) % p == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
                    
            # Add the class
            classes.append([p, best_r])
            
            # Update covered
            for n in range(1, MAX_L + 1):
                if (n - best_r) % p == 0:
                    covered[n] = True
                    
            # Check time
            if time.time() > deadline:
                break
        
        # Find L
        L = 0
        for n in range(1, MAX_L + 1):
            if covered[n]:
                L = n
            else:
                break
                
        return L, classes

    # Candidate 1: Seed
    seed_primes = [p for p in primes if p >= 13]
    L1, c1 = evaluate_primes(seed_primes)
    if L1 > best_L:
        best_L = L1
        best_classes = c1
        
    # Candidate 2: Primes 5..97
    cand2 = [p for p in primes if 5 <= p <= 97]
    L2, c2 = evaluate_primes(cand2)
    if L2 > best_L:
        best_L = L2
        best_classes = c2
        
    # Candidate 3: Primes 3..53
    cand3 = [p for p in primes if 3 <= p <= 53]
    L3, c3 = evaluate_primes(cand3)
    if L3 > best_L:
        best_L = L3
        best_classes = c3
        
    # Candidate 4: Mix of small and large
    # Try to include 2, 3, 5, 7, 11 and some large ones
    # Sum 2,3,5,7,11 = 1.26. Too high.
    # Drop 5: 1.06. Too high.
    # Drop 7: 0.92. OK.
    # Primes: 2, 3, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    cand4 = [2, 3, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    L4, c4 = evaluate_primes(cand4)
    if L4 > best_L:
        best_L = L4
        best_classes = c4
        
    # Candidate 5: 2, 3, 7, 13..97
    # Sum 2,3,7 = 0.97.
    # Add 13 (0.076) -> 1.046. Too high.
    # So only 2, 3, 7.
    cand5 = [2, 3, 7]
    L5, c5 = evaluate_primes(cand5)
    if L5 > best_L:
        best_L = L5
        best_classes = c5
        
    return best_classes
