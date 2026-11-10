import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute prime factors for each number
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        prime_factors[n] = factors

    # Build conflict: two numbers conflict if they share a prime factor
    # For ILP: constraint is for each prime p, sum(x_n for n multiple of p) <= 1

    # 1. Try ILP approach
    best_set = None
    best_sum = 0

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Build constraint matrix A (num_primes x MAX_N)
        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for n in range(1, MAX_N + 1):
            for p in prime_factors[n]:
                A[prime_to_idx[p], n - 1] = 1

        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)

        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            if len(selected) > 0:
                # Verify validity
                valid = True
                sel_set = set(selected)
                for a in selected:
                    for b in selected:
                        if a < b and gcd(a, b) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_set = sel_set
                    best_sum = sum(selected)
    except Exception:
        pass

    # 2. If ILP didn't give good result, use optimized prime-power selection + local search
    if best_set is None or best_sum < 1000:
        # Key insight: for small primes, we can combine their prime powers into products
        # For primes p, q: if p^a * q^b <= 256, we can use that product instead of separate powers
        # This is a maximum-weight matching problem on small primes
        
        # Find max prime power for each prime <= 256
        max_prime_power = {}
        for p in primes:
            power = p
            while power * p <= 256:
                power *= p
            max_prime_power[p] = power
        
        # Small primes that can potentially be merged
        small_primes = [p for p in primes if p <= 16]  # 2, 3, 5, 7, 11, 13
        small_prime_set = set(small_primes)
        
        # For each pair of small primes, check if their product of max powers fits
        # and compute the gain over using them separately
        def get_best_small_prime_assignment():
            """Find optimal assignment using matching for small primes"""
            n_sp = len(small_primes)
            if n_sp == 0:
                return {p: max_prime_power[p] for p in small_primes}, sum(max_prime_power[p] for p in small_primes)
            
            # Weight matrix for matching: weight[i,j] = gain of merging small_primes[i] and small_primes[j]
            # If we merge p and q, we get p^a * q^b instead of p^a + q^b
            # Gain = p^a * q^b - p^a - q^b
            weight = np.zeros((n_sp, n_sp))
            merged_value = np.zeros((n_sp, n_sp))
            
            for i in range(n_sp):
                for j in range(i+1, n_sp):
                    p, q = small_primes[i], small_primes[j]
                    pp, qp = max_prime_power[p], max_prime_power[q]
                    product = pp * qp
                    if product <= 256:
                        # We can merge them
                        merged_value[i, j] = product
                        merged_value[j, i] = product
                        weight[i, j] = product - pp - qp
                        weight[j, i] = weight[i, j]
            
            # Now solve: maximize sum of selected values
            # Each small prime can be either:
            # - Alone: value = max_prime_power[p]
            # - Merged with another: value = merged_value (shared between the two)
            # This is a maximum weight matching where unmatched nodes get their base value
            
            # Brute force for small number of primes (6 primes = 15 ways to pair)
            best_val = sum(max_prime_power[p] for p in small_primes)
            best_assignment = {p: max_prime_power[p] for p in small_primes}
            
            # Try all matchings
            def try_matching(remaining, current_pairs, current_val):
                nonlocal best_val, best_assignment
                if not remaining:
                    if current_val > best_val:
                        best_val = current_val
                        # Reconstruct assignment
                        new_assign = {}
                        for idx, p in enumerate(small_primes):
                            new_assign[p] = max_prime_power[p]
                        for (i, j) in current_pairs:
                            new_assign[small_primes[i]] = merged_value[i, j]
                            new_assign[small_primes[j]] = merged_value[i, j]
                        best_assignment = dict(new_assign)
                    return
                
                # Option 1: leave smallest remaining alone
                first = remaining[0]
                try_matching(remaining[1:], current_pairs, current_val + max_prime_power[small_primes[first]])
                
                # Option 2: pair first with another
                for k in range(1, len(remaining)):
                    second = remaining[k]
                    if merged_value[first, second] > 0:
                        new_remaining = remaining[1:k] + remaining[k+1:]
                        try_matching(new_remaining, current_pairs + [(first, second)], current_val + merged_value[first, second])
            
            try_matching(list(range(n_sp)), [], 0)
            
            return best_assignment, best_val
        
        small_assignment, small_sum = get_best_small_prime_assignment()
        
        # Build initial set: 1 + small prime assignments + large prime powers
        S = {1}
        for p in small_primes:
            S.add(small_assignment[p])
        
        # Add prime powers for larger primes
        for p in primes:
            if p > 16:
                S.add(max_prime_power[p])
        
        # Verify validity
        valid = True
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if gcd(S_list[i], S_list[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_set = S
            best_sum = sum(S)
        else:
            # Fallback to greedy
            S = {1}
            for n in range(256, 1, -1):
                if all(gcd(n, x) == 1 for x in S):
                    S.add(n)
            best_set = S
            best_sum = sum(S)

    # 3. Local search: try swaps and additions
    def is_valid(set_to_check):
        lst = list(set_to_check)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    # Enhanced: try to merge pairs of small primes' contributions
    # If we have a and b in best_set where a = p^x, b = q^y, and p^x * q^y <= 256
    # and p^x * q^y > a + b, then replace with the product
    def try_merge_pairs():
        nonlocal best_set, best_sum
        improved = False
        S_list = list(best_set)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if a == 1 or b == 1:
                    continue
                product = a * b
                if product > 256:
                    continue
                if product <= max(a, b):
                    continue
                # Check if product is compatible with all other elements
                ok = True
                for x in best_set:
                    if x != a and x != b and gcd(product, x) != 1:
                        ok = False
                        break
                if ok:
                    new_sum = best_sum - a - b + product
                    if new_sum > best_sum:
                        best_set = (best_set - {a, b}) | {product}
                        best_sum = new_sum
                        improved = True
        return improved

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try merging pairs first (most impactful)
        if time.time() < deadline:
            if try_merge_pairs():
                improved = True
                continue
        S_list = sorted(best_set, reverse=True)

        # Try 1-for-1 swaps: remove a, add b if b > a and compatible
        for a in list(S_list):
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                # Check if b is compatible with all elements except a
                ok = True
                for x in best_set:
                    if x != a and gcd(b, x) != 1:
                        ok = False
                        break
                if ok:
                    new_set = (best_set - {a}) | {b}
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set = new_set
                        best_sum = new_sum
                        improved = True
                        break

        # Try adding a number if it fits
        if time.time() < deadline:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break

        # Try 2-for-1 swaps: remove two small numbers, add one large
        if time.time() < deadline and len(best_set) >= 2:
            S_sorted = sorted(best_set)
            for i in range(len(S_sorted)):
                a = S_sorted[i]
                if a == 1:
                    continue
                for j in range(i + 1, len(S_sorted)):
                    b = S_sorted[j]
                    if b == 1:
                        continue
                    if a + b < 256:
                        continue  # need to add something larger
                    for c in range(256, max(a, b), -1):
                        if c in best_set:
                            continue
                        # Check c compatible with all except a and b
                        ok = True
                        for x in best_set:
                            if x != a and x != b and gcd(c, x) != 1:
                                ok = False
                                break
                        if ok:
                            new_sum = best_sum - a - b + c
                            if new_sum > best_sum:
                                best_set = (best_set - {a, b}) | {c}
                                best_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Final verification
    result = sorted(best_set)
    if not result:
        result = [1]

    # Ensure all are Python ints
    return [int(x) for x in result]
