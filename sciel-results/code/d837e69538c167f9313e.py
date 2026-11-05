import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    primes_set = set(primes)

    # Helper: Calculate the coverage length L for a given set of classes
    # Returns the largest L such that 1..L are all covered.
    def calc_coverage_length(classes, max_check=100000):
        if not classes:
            return 0
        
        current_limit = 20000
        while True:
            covered = np.zeros(current_limit + 1, dtype=bool)
            for p, r in classes:
                start = p if r == 0 else r
                if start <= current_limit:
                    idx = np.arange(start, current_limit + 1, p)
                    covered[idx] = True
            
            # Find first gap
            uncovered = np.where(~covered[1:])[0]
            if len(uncovered) == 0:
                # All 1..current_limit covered
                if current_limit >= 100000:
                    return 100000
                current_limit = min(current_limit * 2, 100000)
                if current_limit == 100000:
                    # One last check at the cap
                    covered = np.zeros(100001, dtype=bool)
                    for p, r in classes:
                        start = p if r == 0 else r
                        if start <= 100000:
                            idx = np.arange(start, 100001, p)
                            covered[idx] = True
                    uncovered = np.where(~covered[1:])[0]
                    if len(uncovered) == 0:
                        return 100000
                    else:
                        return uncovered[0]
            else:
                return uncovered[0]

    # Helper: Greedy assignment of residues for a fixed set of primes
    def greedy_assign(primes_set, max_check=20000):
        primes_sorted = sorted(primes_set)
        residues = {}
        covered = np.zeros(max_check + 1, dtype=bool)
        
        for p in primes_sorted:
            best_r = 0
            best_gain = -1
            
            for r in range(p):
                start = p if r == 0 else r
                if start > max_check:
                    continue
                idx = np.arange(start, max_check + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[p] = best_r
            start = p if best_r == 0 else best_r
            if start <= max_check:
                idx = np.arange(start, max_check + 1, p)
                covered[idx] = True
        
        return [[p, residues[p]] for p in primes_sorted]

    # Helper: Fast local search
    def fast_local_search(classes, max_iter=100, max_check=20000):
        current_classes = [list(c) for c in classes]
        current_L = calc_coverage_length(current_classes, max_check)
        
        if current_L == 0:
            return current_classes, 0
            
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            for i in range(len(current_classes)):
                p, r = current_classes[i]
                best_r = r
                best_L = current_L
                
                for r_new in range(p):
                    if r_new == r:
                        continue
                    
                    current_classes[i][1] = r_new
                    new_L = calc_coverage_length(current_classes, max_check)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_new
                    else:
                        current_classes[i][1] = r
                
                if best_r != r:
                    current_classes[i][1] = best_r
                    current_L = best_L
                    improved = True
                    
        return current_classes, current_L

    # Helper: Check if a set of primes fits the budget
    def fits_budget(prime_list):
        s = Fraction(0)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    # Main Strategy:
    # 1. Generate diverse candidate sets of primes that fit the budget.
    # 2. For each, run greedy + local search.
    # 3. Keep the best.
    
    best_classes = []
    best_L = 0

    # Candidate Generation
    # We want sets with high density (small primes) but enough primes to cover gaps.
    # Sum 1/p <= 1.
    # 1/2 + 1/3 + 1/5 + 1/7 = 0.5 + 0.333 + 0.2 + 0.142 = 1.175 > 1.
    # So we can't have {2,3,5,7}.
    # {2,3,7} = 0.975 <= 1.
    # {2,5,7} = 0.842 <= 1.
    # {3,5,7} = 0.675 <= 1.
    # {2,3,11} = 0.863 <= 1.
    # {2,3,13} = 0.826 <= 1.
    
    # Let's try to build sets by adding primes one by one to a base, or removing from a full set.
    
    # Base: All primes >= 11. Sum is small.
    # We can add some small primes if they fit.
    
    candidates = []
    
    # 1. All primes >= 11
    c1 = [p for p in primes if p >= 11]
    if fits_budget(c1):
        candidates.append(c1)
        
    # 2. All primes >= 13
    c2 = [p for p in primes if p >= 13]
    if fits_budget(c2):
        candidates.append(c2)
        
    # 3. Try to add small primes to c2
    small_primes = [2, 3, 5, 7]
    for sp in small_primes:
        c3 = [sp] + [p for p in primes if p >= 13]
        if fits_budget(c3):
            candidates.append(c3)
            
    # 4. Pairs of small primes + large primes
    for i in range(len(small_primes)):
        for j in range(i+1, len(small_primes)):
            c4 = [small_primes[i], small_primes[j]] + [p for p in primes if p >= 17]
            if fits_budget(c4):
                candidates.append(c4)
                
    # 5. Specific known good sets
    c5 = [2, 3, 7] + [p for p in primes if p >= 17]
    if fits_budget(c5):
        candidates.append(c5)
        
    c6 = [2, 5, 7] + [p for p in primes if p >= 17]
    if fits_budget(c6):
        candidates.append(c6)
        
    c7 = [3, 5, 7] + [p for p in primes if p >= 17]
    if fits_budget(c7):
        candidates.append(c7)

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    # Evaluate candidates
    max_check = 20000
    
    for i, cand_primes in enumerate(unique_candidates):
        if time.time() > deadline - 10:
            break
            
        # Greedy Assignment
        classes = greedy_assign(cand_primes, max_check)
        
        # Local Search
        classes, L = fast_local_search(classes, max_iter=50, max_check=max_check)
        
        if L > best_L:
            best_L = L
            best_classes = classes
            
    # If no good candidate found, fallback
    if best_L == 0:
        test_classes = [[2, 1], [3, 0]]
        L = calc_coverage_length(test_classes, 1000)
        if L > best_L:
            best_L = L
            best_classes = test_classes

    # Final Verification
    if best_classes:
        # Quick check
        if fits_budget([p for p, r in best_classes]):
            return best_classes
            
    return [[2, 1]]
