import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s kill

    # Fallback: known valid set of 16 elements
    fallback_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)

    # Precompute exponent vectors for all integers 2..40
    def factorize(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return vec

    # Map: exponent vector tuple -> list of integers that have this vector
    vec_to_nums = {}
    for n in range(2, 41):
        v = tuple(factorize(n))
        if v not in vec_to_nums:
            vec_to_nums[v] = []
        vec_to_nums[v].append(n)

    # Candidate exponent vectors (unique ones)
    unique_vecs = list(vec_to_nums.keys())

    # Compute max possible exponent sum per prime across ALL candidates 2..40
    max_exp_sum = [0] * num_primes
    for v in unique_vecs:
        for i in range(num_primes):
            max_exp_sum[i] += v[i]
    
    # Use tight bases for encoding
    bases = [max_exp_sum[i] + 1 for i in range(num_primes)]

    def encode_vector(vec):
        code = 0
        for i in range(num_primes):
            code = code * bases[i] + vec[i]
        return code

    # Precompute encoded form of each unique vector
    vec_codes = [(v, encode_vector(v)) for v in unique_vecs]

    # Optimized compatibility check using bitset-like approach
    def is_compatible_fast(sums_set, new_code):
        # For each s in sums_set, check if s + new_code is in sums_set
        # This is equivalent to: (sums_set + new_code) & sums_set == empty set
        # But we can't do set arithmetic directly, so iterate
        for s in sums_set:
            if s + new_code in sums_set:
                return False
        return True

    def add_to_sums_fast(sums_set, new_code):
        return sums_set | {s + new_code for s in sums_set}

    # Try multiple orderings to find good solutions quickly
    # Ordering 1: by sum of exponents (ascending) - prefer "small" vectors
    # Ordering 2: by number of distinct primes (ascending) - prefer sparse vectors
    # Ordering 3: by max exponent (ascending)
    
    def get_ordering(order_type):
        if order_type == 0:
            # Sort by sum of exponents, then lexicographically
            return sorted(vec_codes, key=lambda x: (sum(x[0]), x[0]))
        elif order_type == 1:
            # Sort by number of nonzero components, then sum
            return sorted(vec_codes, key=lambda x: (sum(1 for e in x[0] if e > 0), sum(x[0]), x[0]))
        elif order_type == 2:
            # Sort by max exponent, then sum
            return sorted(vec_codes, key=lambda x: (max(x[0]) if x[0] else 0, sum(x[0]), x[0]))
        elif order_type == 3:
            # Reverse of ordering 0: larger vectors first
            return sorted(vec_codes, key=lambda x: (-sum(x[0]), x[0]))
        else:
            return sorted(vec_codes, key=lambda x: (sum(x[0]), x[0]))

    best_count = 0
    best_vecs = []

    def greedy_search(ordered_vecs):
        nonlocal best_count, best_vecs
        sums = {0}
        chosen = []
        for v, code in ordered_vecs:
            if is_compatible_fast(sums, code):
                chosen.append(v)
                sums = add_to_sums_fast(sums, code)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_vecs = list(chosen)
        return chosen

    # Run greedy with different orderings to get good initial lower bound
    for ot in range(4):
        if time.time() > deadline:
            break
        ordered = get_ordering(ot)
        greedy_search(ordered)

    # DFS with branch and bound
    def dfs(idx, current_vecs, current_sums, ordered_vecs, remaining_count):
        nonlocal best_count, best_vecs, deadline

        if time.time() > deadline:
            return

        if len(current_vecs) > best_count:
            best_count = len(current_vecs)
            best_vecs = list(current_vecs)

        if idx >= len(ordered_vecs):
            return
        
        # Prune: even if we add all remaining, can't beat best
        if len(current_vecs) + remaining_count <= best_count:
            return

        v, code = ordered_vecs[idx]
        
        # Try to include
        if is_compatible_fast(current_sums, code):
            new_sums = add_to_sums_fast(current_sums, code)
            current_vecs.append(v)
            dfs(idx + 1, current_vecs, new_sums, ordered_vecs, remaining_count - 1)
            current_vecs.pop()

        # Try to exclude
        dfs(idx + 1, current_vecs, current_sums, ordered_vecs, remaining_count - 1)

    # Try DFS with the ordering that gave the best greedy result
    # First, find which ordering gave the best greedy result
    greedy_results = {}
    for ot in range(4):
        if time.time() > deadline:
            break
        ordered = get_ordering(ot)
        result = greedy_search(ordered)
        greedy_results[ot] = (len(result), ot)

    # Sort orderings by their greedy performance (best first)
    sorted_orderings = sorted(greedy_results.keys(), key=lambda ot: -greedy_results[ot][0])

    # Run DFS with the best-performing ordering
    for ot in sorted_orderings:
        if time.time() > deadline:
            break
        ordered = get_ordering(ot)
        n = len(ordered)
        dfs(0, [], {0}, ordered, n)

    # Map best_vecs back to integers
    result = []
    for v in best_vecs:
        nums = vec_to_nums[v]
        if nums:
            result.append(nums[0])

    # Verify the result
    def verify(nums):
        if len(nums) < 2 or len(nums) > 18:
            return False
        if any(not isinstance(x, int) for x in nums):
            return False
        if any(x < 2 or x > 40 for x in nums):
            return False
        if len(set(nums)) != len(nums):
            return False
        # Check all subset products are distinct
        products = set()
        for r in range(len(nums) + 1):
            for comb in combinations(nums, r):
                prod = 1
                for x in comb:
                    prod *= x
                if prod in products:
                    return False
                products.add(prod)
        return True

    if not verify(result):
        result = fallback_set

    return result
