import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute: for each number, its prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Precompute prime factors for all numbers 1-256 (max sum is 128+128=256)
    pf_cache = {i: prime_factors(i) for i in range(1, 257)}

    def is_pairwise_coprime(nums):
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate_fast(A, B):
        """Fast evaluation using precomputed prime factors."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = list(sums)
        n = len(slist)
        for i in range(n):
            for j in range(i + 1, n):
                if pf_cache[slist[i]] & pf_cache[slist[j]]:
                    return 0
        return n

    # Precompute a "coprime compatibility" table for sums
    # compat[s1][s2] = True if gcd(s1, s2) == 1
    max_sum = 256
    compat = [[False] * (max_sum + 1) for _ in range(max_sum + 1)]
    for s1 in range(2, max_sum + 1):
        for s2 in range(s1, max_sum + 1):
            c = (math.gcd(s1, s2) == 1)
            compat[s1][s2] = c
            compat[s2][s1] = c

    def evaluate_compat(A, B):
        """Evaluation using precomputed compatibility table."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = list(sums)
        n = len(slist)
        for i in range(n):
            for j in range(i + 1, n):
                if not compat[slist[i]][slist[j]]:
                    return 0
        return n

    # Find initial best
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_cardinality = evaluate_compat(best_A, best_B)
    if best_cardinality == 0:
        # Fallback to a known valid small config
        best_A = [1, 2, 3]
        best_B = [4, 7, 11]
        best_cardinality = evaluate_compat(best_A, best_B)

    def local_search(A, B, max_iter=20000):
        """Local search with smarter move ordering."""
        A = list(A)
        B = list(B)
        curr_score = evaluate_compat(A, B)
        best_A = list(A)
        best_B = list(B)
        best_score = curr_score

        # Precompute current sums
        def get_sums(A, B):
            s = set()
            for a in A:
                for b in B:
                    s.add(a + b)
            return s

        # Try all single-element replacements, ordered by potential
        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            improved = False
            current_sums = get_sums(A, B)

            # Try replacing each element in A
            for pos in range(3):
                old_val = A[pos]
                # Try candidate values sorted by how many new sums they create
                # that are coprime with existing sums
                candidates = []
                for new_val in range(1, 129):
                    if new_val in A:
                        continue
                    # Compute new sums
                    new_sums = set()
                    for b in B:
                        new_sums.add(new_val + b)
                    # Count how many new sums are coprime with all current sums
                    # (excluding the old sums that will be removed)
                    removed_sums = set()
                    for b in B:
                        removed_sums.add(old_val + b)
                    
                    # New distinct sums = (current_sums - removed_sums) | new_sums
                    # Check if all are pairwise coprime
                    all_new_sums = (current_sums - removed_sums) | new_sums
                    al = list(all_new_sums)
                    valid = True
                    for i in range(len(al)):
                        for j in range(i + 1, len(al)):
                            if not compat[al[i]][al[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        score = len(all_new_sums)
                        candidates.append((score, new_val))
                
                # Sort by score descending, then by value
                candidates.sort(key=lambda x: (-x[0], x[1]))
                
                for score, new_val in candidates:
                    if score > curr_score:
                        A[pos] = new_val
                        curr_score = score
                        improved = True
                        if curr_score > best_score:
                            best_score = curr_score
                            best_A = list(A)
                            best_B = list(B)
                        if curr_score == 9:
                            return best_A, best_B, best_score
                        break
                if improved:
                    break

            if improved:
                continue

            # Try replacing each element in B
            for pos in range(3):
                old_val = B[pos]
                candidates = []
                for new_val in range(1, 129):
                    if new_val in B:
                        continue
                    new_sums = set()
                    for a in A:
                        new_sums.add(a + new_val)
                    removed_sums = set()
                    for a in A:
                        removed_sums.add(a + old_val)
                    
                    all_new_sums = (current_sums - removed_sums) | new_sums
                    al = list(all_new_sums)
                    valid = True
                    for i in range(len(al)):
                        for j in range(i + 1, len(al)):
                            if not compat[al[i]][al[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        score = len(all_new_sums)
                        candidates.append((score, new_val))
                
                candidates.sort(key=lambda x: (-x[0], x[1]))
                
                for score, new_val in candidates:
                    if score > curr_score:
                        B[pos] = new_val
                        curr_score = score
                        improved = True
                        if curr_score > best_score:
                            best_score = curr_score
                            best_A = list(A)
                            best_B = list(B)
                        if curr_score == 9:
                            return best_A, best_B, best_score
                        break
                if improved:
                    break

            if not improved:
                # Try a random perturbation to escape local optimum
                if random.random() < 0.3:
                    # Replace a random element with a random value
                    if random.random() < 0.5:
                        pos = random.randint(0, 2)
                        new_val = random.randint(1, 128)
                        if new_val not in A:
                            A[pos] = new_val
                    else:
                        pos = random.randint(0, 2)
                        new_val = random.randint(1, 128)
                        if new_val not in B:
                            B[pos] = new_val
                    curr_score = evaluate_compat(A, B)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_A = list(A)
                        best_B = list(B)
                else:
                    break

        return best_A, best_B, best_score

    # Multiple random starts with diverse ranges
    random.seed(42)
    
    # Define several ranges to sample from
    ranges = [
        (1, 30),
        (1, 60),
        (1, 100),
        (1, 128),
        (5, 50),
        (10, 80),
    ]
    
    start_idx = 0
    total_starts = 0
    while time.time() < deadline:
        rng = ranges[start_idx % len(ranges)]
        A = random.sample(range(rng[0], rng[1] + 1), 3)
        B = random.sample(range(rng[0], rng[1] + 1), 3)
        start_idx += 1
        total_starts += 1
        
        res_A, res_B, res_score = local_search(A, B, max_iter=10000)
        
        if res_score > best_cardinality:
            best_A = res_A
            best_B = res_B
            best_cardinality = res_score
            if res_score == 9:
                break

    return [best_A, best_B]
