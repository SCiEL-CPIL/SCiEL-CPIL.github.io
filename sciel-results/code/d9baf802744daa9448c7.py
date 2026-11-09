import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    # Sieve primes up to 256 (max sum = 128+128)
    max_sum = 256
    is_prime_arr = [True] * (max_sum + 1)
    is_prime_arr[0] = False
    is_prime_arr[1] = False
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime_arr[j] = False

    def is_prime(n):
        return 0 <= n <= max_sum and is_prime_arr[n]

    def is_pairwise_coprime(nums):
        if len(nums) < 2:
            return True
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    # Structural change: direct exhaustive search over all 3-element subsets
    # of [1..128] for A and B, but with smart pruning.
    # Total combos: C(128,3)^2 ≈ 44M^2 — too many.
    # Instead: iterate over all A (C(128,3) ≈ 343k), and for each A,
    # find B that maximizes distinct pairwise-coprime sums.
    # For each A, the 3 elements are fixed. For each B candidate,
    # compute the 9 sums and check.
    #
    # Better structural approach: iterate over all possible A (343k),
    # then for each A, iterate over all possible B (343k) but with early
    # termination. 343k * 343k is too much.
    #
    # Alternative: Since we want 9 distinct pairwise-coprime sums (ideally primes),
    # we can search over pairs of 3-element sets more efficiently by:
    # 1. Precompute all 3-element sets
    # 2. For each pair, check if all 9 sums are pairwise coprime
    # 3. But 343k^2 is ~118 billion — too slow
    #
    # Better: Fix A, then for each B, the sums are {a_i + b_j}.
    # We can precompute for each A the set of all possible sums with any b in [1,128].
    # Actually, let's just do a smarter search:
    #
    # Key insight: if all 9 sums are prime, they're automatically pairwise coprime.
    # So search for A, B such that all a_i + b_j are prime.
    #
    # Strategy: iterate over all A (sorted), then for each A, find B such that
    # all 9 sums are prime. For a fixed A = {a1, a2, a3}, we need b1, b2, b3
    # such that a_i + b_j is prime for all i, j.
    #
    # For each candidate b, check which a_i + b are prime.
    # We need 3 b's such that for each b_j, all a_i + b_j are prime.
    #
    # Precompute for each value v in [1,128] and each a in [1,128] whether a+v is prime.
    # Then for a fixed A, a valid b must satisfy: is_prime(a1+b) and is_prime(a2+b) and is_prime(a3+b).
    # Collect all valid b's, then pick 3 distinct ones.
    #
    # This is O(C(128,3) * 128) for finding valid b's per A, which is ~343k * 128 ≈ 44M — feasible.

    # Precompute prime sums table
    # prime_sum[a][b] = is_prime(a+b) for a,b in 1..128
    # We'll use the sieve array directly.

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B)

    # Generate all 3-element subsets of [1..128]
    # There are C(128,3) = 343,776 of them
    all_A = list(combinations(range(1, 129), 3))

    for idx, A in enumerate(all_A):
        if time.time() > deadline:
            break

        # Find all b in [1,128] such that a1+b, a2+b, a3+b are all prime
        valid_bs = []
        a1, a2, a3 = A
        for b in range(1, 129):
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if s1 <= max_sum and s2 <= max_sum and s3 <= max_sum:
                if is_prime_arr[s1] and is_prime_arr[s2] and is_prime_arr[s3]:
                    valid_bs.append(b)

        # We need 3 distinct b's from valid_bs
        if len(valid_bs) >= 3:
            # Try all combinations of 3 from valid_bs
            # But valid_bs could be large. Let's try the first few.
            # To maximize distinct sums, we want b's that are spread out.
            # Actually, any 3 distinct b's will give 9 sums.
            # Let's just pick the first 3 and verify.
            B = [valid_bs[0], valid_bs[1], valid_bs[2]]
            score = evaluate(A, B)
            if score == 9:
                return [list(A), B]
            # Try other combinations if needed
            # But first try: pick b's that are far apart to maximize distinct sums
            # Actually all 9 sums are prime so they're distinct if A and B elements are distinct
            # which they are. So score should be 9.
            # Let me just verify and return.

    # If prime-based search failed, do a broader search for pairwise coprime (not necessarily prime)
    # This is harder. Let's try a different structural approach:
    # Search over A, then for each A, search over B with pairwise coprime check.
    # But this is expensive. Let's limit to a subset.

    # Fallback: try a random/smart search
    import random
    random.seed(42)
    for trial in range(100000):
        if time.time() > deadline:
            break
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if score == 9:
                return [A, B]

    return [best_A, best_B]
