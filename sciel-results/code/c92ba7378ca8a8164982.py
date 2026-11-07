import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    all_nums = list(range(2, 41))

    # Precompute exponent vectors for all numbers 2..40
    # Each number n is represented as a tuple of exponents over the primes
    vecs = {}
    for n in all_nums:
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vecs[n] = tuple(v)

    zero = (0,) * num_primes

    def check_addition(current_sums, vec):
        """
        Check if adding a vector with exponent vector 'vec' keeps all subset sums distinct.
        current_sums: set of all current subset sums (as tuples)
        Returns: (is_valid, new_sums_to_add)
        """
        new_sums = []
        for s in current_sums:
            ns = tuple(s[i] + vec[i] for i in range(num_primes))
            if ns in current_sums:
                return False, None
            new_sums.append(ns)
        # Also need to check that new_sums are distinct among themselves
        # Since s are distinct and vec is fixed, ns are distinct for distinct s.
        # So we only need to check against current_sums.
        return True, new_sums

    # Initialize with primes
    best_set = primes[:]
    best_len = len(primes)

    # Verify initial set and build sums
    current_sums = {zero}
    valid_initial = True
    for p in primes:
        v = vecs[p]
        is_valid, new_sums = check_addition(current_sums, v)
        if not is_valid:
            valid_initial = False
            break
        current_sums.update(new_sums)

    if not valid_initial:
        # This should never happen for primes, but fallback
        return [int(x) for x in primes]

    # Candidates: composites in 2..40
    composite_set = set(all_nums) - set(primes)
    candidates = list(composite_set)

    # Deterministic greedy first: try adding composites in sorted order
    S = primes[:]
    sums = {zero}
    for p in primes:
        v = vecs[p]
        _, new_sums = check_addition(sums, v)
        sums.update(new_sums)

    for n in sorted(candidates):
        if time.time() > deadline:
            break
        v = vecs[n]
        is_valid, new_sums = check_addition(sums, v)
        if is_valid:
            S.append(n)
            sums.update(new_sums)

    if len(S) > best_len:
        best_len = len(S)
        best_set = S[:]

    # Randomized greedy restarts
    random.seed(42)  # Reproducibility
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        shuffled = candidates[:]
        random.shuffle(shuffled)

        S = primes[:]
        sums = {zero}
        for p in primes:
            v = vecs[p]
            _, new_sums = check_addition(sums, v)
            sums.update(new_sums)

        for n in shuffled:
            if time.time() > deadline:
                break
            v = vecs[n]
            is_valid, new_sums = check_addition(sums, v)
            if is_valid:
                S.append(n)
                sums.update(new_sums)

        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
            if best_len >= 18:
                break

    # Final verification of best_set to ensure correctness
    def verify_distinct_subset_products(nums):
        """Verify that all 2^k subset products are distinct."""
        k = len(nums)
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True

    if not verify_distinct_subset_products(best_set):
        # Fallback to primes
        return [int(x) for x in primes]

    return [int(x) for x in best_set]
