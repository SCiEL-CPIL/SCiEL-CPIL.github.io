import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 142  # Safety margin before 150s kill

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

    # Helper: Calculate coverage array for a set of classes
    def compute_coverage(classes, limit):
        covered = np.zeros(limit + 1, dtype=bool)
        for p, r in classes:
            start = p if r == 0 else r
            if start <= limit:
                idx = np.arange(start, limit + 1, p)
                covered[idx] = True
        return covered

    # Helper: Find first gap in coverage
    def find_first_gap(covered, limit):
        uncovered = np.where(~covered[1:limit + 1])[0]
        if len(uncovered) == 0:
            return limit
        return uncovered[0]

    # Helper: Calculate coverage length
    def calc_coverage_length(classes, initial_limit=5000, max_check=100000):
        if not classes:
            return 0
        limit = initial_limit
        while True:
            covered = compute_coverage(classes, limit)
            gap = find_first_gap(covered, limit)
            if gap >= limit:
                if limit >= max_check:
                    return max_check
                limit = min(limit * 2, max_check)
            else:
                return gap

    # Helper: Fast local search using incremental coverage updates
    def fast_local_search(classes, max_iter=80, check_limit=10000):
        current_classes = [list(c) for c in classes]
        n = len(current_classes)
        
        # Initial coverage
        covered = compute_coverage(current_classes, check_limit)
        current_L = find_first_gap(covered, check_limit)
        
        if current_L == 0:
            return current_classes, 0
            
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            
            for i in range(n):
                p, r = current_classes[i]
                best_r = r
                best_L = current_L
                
                for r_new in range(p):
                    if r_new == r:
                        continue
                    
                    # Remove old residue's coverage
                    old_start = p if r == 0 else r
                    if old_start <= check_limit:
                        old_idx = np.arange(old_start, check_limit + 1, p)
                        # Only remove if this prime's coverage is the only one
                        # For simplicity, recompute affected region
                        pass
                    
                    # Set new residue
                    current_classes[i][1] = r_new
                    
                    # Recompute coverage (fast for small check_limit)
                    covered = compute_coverage(current_classes, check_limit)
                    new_L = find_first_gap(covered, check_limit)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_new
                    else:
                        current_classes[i][1] = r
                
                if best_r != r:
                    current_classes[i][1] = best_r
                    current_L = best_L
                    improved = True
                    
        # Final full check
        final_L = calc_coverage_length(current_classes, 5000, 100000)
        return current_classes, final_L

    # Helper: Check if a set of primes fits the budget
    def fits_budget(prime_list):
        s = Fraction(0)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    # Helper: Greedy assignment
    def greedy_assign(prime_list, max_check=10000):
        primes_sorted = sorted(prime_list)
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

    # Main Strategy
    best_classes = []
    best_L = 0

    # Generate diverse candidate sets
    candidates = []
    small_primes = [2, 3, 5, 7]
    
    # Base sets with different minimum primes
    for min_p in [11, 13, 17, 19, 23, 29]:
        c = [p for p in primes if p >= min_p]
        if fits_budget(c):
            candidates.append(c)
    
    # Add single small primes to base sets
    for sp in small_primes:
        for min_p in [13, 17, 19, 23]:
            c = [sp] + [p for p in primes if p >= min_p]
            if fits_budget(c):
                candidates.append(c)
    
    # Pairs of small primes
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            for min_p in [17, 19, 23, 29]:
                c = [small_primes[i], small_primes[j]] + [p for p in primes if p >= min_p]
                if fits_budget(c):
                    candidates.append(c)
    
    # Triples of small primes (if budget allows)
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            for k in range(j + 1, len(small_primes)):
                c = [small_primes[i], small_primes[j], small_primes[k]] + [p for p in primes if p >= 29]
                if fits_budget(c):
                    candidates.append(c)
    
    # Specific known good combinations
    specific = [
        [2, 3, 7] + [p for p in primes if p >= 17],
        [2, 5, 7] + [p for p in primes if p >= 17],
        [3, 5, 7] + [p for p in primes if p >= 17],
        [2, 3, 11] + [p for p in primes if p >= 17],
        [2, 5, 11] + [p for p in primes if p >= 17],
        [3, 5, 11] + [p for p in primes if p >= 17],
        [2, 3, 13] + [p for p in primes if p >= 17],
        [2, 7, 11] + [p for p in primes if p >= 17],
        [3, 7, 11] + [p for p in primes if p >= 17],
        [5, 7, 11] + [p for p in primes if p >= 17],
        [2, 3, 7, 13] if fits_budget([2, 3, 7, 13] + [p for p in primes if p >= 19]) else [],
    ]
    for c in specific:
        if c and fits_budget(c):
            candidates.append(c)

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Sort candidates by potential (prefer those with more small primes)
    def candidate_score(c):
        small_count = sum(1 for p in c if p <= 13)
        return -small_count  # Negative for sorting (more small primes = higher priority)
    
    unique_candidates.sort(key=candidate_score)

    # Evaluate candidates
    max_check = 10000
    
    for i, cand_primes in enumerate(unique_candidates):
        if time.time() > deadline - 12:
            break
        
        # Greedy Assignment
        classes = greedy_assign(cand_primes, max_check)
        
        # Local Search
        classes, L = fast_local_search(classes, max_iter=60, check_limit=max_check)
        
        if L > best_L:
            best_L = L
            best_classes = classes

    # If no good candidate found, fallback
    if best_L == 0:
        test_classes = [[2, 1], [3, 0]]
        L = calc_coverage_length(test_classes, 1000, 5000)
        if L > best_L:
            best_L = L
            best_classes = test_classes

    # Final Verification
    if best_classes:
        if fits_budget([p for p, r in best_classes]):
            return best_classes
            
    return [[2, 1]]
