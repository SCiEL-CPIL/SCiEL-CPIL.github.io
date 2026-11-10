"""Combined: prime-intersection construction + structured/random search for pairwise-coprime sumsets."""
import math
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit

    best_cardinality = 0
    best_result = [[1, 7, 13], [4, 10, 16]]

    # Precompute primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(256)
    primes_set = set(x for x in range(2, 257) if is_prime[x])

    def check_pairwise_coprime(values):
        """Check if all distinct values in the set are pairwise coprime."""
        distinct_vals = list(set(values))
        n = len(distinct_vals)
        for i in range(n):
            vi = distinct_vals[i]
            for j in range(i + 1, n):
                if gcd(vi, distinct_vals[j]) != 1:
                    return False
        return True

    def try_pair(A, B):
        nonlocal best_cardinality, best_result
        A_sorted = sorted(A)
        B_sorted = sorted(B)
        if len(set(A_sorted)) != 3 or len(set(B_sorted)) != 3:
            return False
        if any(x < 1 or x > 128 for x in A_sorted) or any(x < 1 or x > 128 for x in B_sorted):
            return False

        sums = [a + b for a in A_sorted for b in B_sorted]
        distinct_sums = set(sums)

        if len(distinct_sums) <= best_cardinality:
            return False

        if check_pairwise_coprime(list(distinct_sums)):
            best_cardinality = len(distinct_sums)
            best_result = [A_sorted[:], B_sorted[:]]
            return True
        return False

    # ---- Strategy 1 (from inspiration 2): prime-intersection construction ----
    # For a given A = [a1,a2,a3], find all b in [1,128] such that a1+b, a2+b, a3+b are all prime.
    # If we can find >=3 such b's, pick 3 and we get all 9 sums prime (hence pairwise coprime)
    # and (ideally) all 9 distinct.
    for a1 in range(1, 50):
        if time.time() > deadline:
            return best_result
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline:
                return best_result
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline:
                    return best_result
                A = [a1, a2, a3]
                valid_b = []
                for b in range(1, 129):
                    if (a1 + b) in primes_set and (a2 + b) in primes_set and (a3 + b) in primes_set:
                        valid_b.append(b)
                if len(valid_b) < 3:
                    continue
                # Try all triples from valid_b
                for b1, b2, b3 in combinations(valid_b, 3):
                    B = [b1, b2, b3]
                    try_pair(A, B)
                    if best_cardinality >= 9:
                        return best_result

    if time.time() > deadline:
        return best_result

    # ---- Strategy 2 (from current): arithmetic progressions ----
    for a in range(1, 80):
        if time.time() > deadline:
            return best_result
        for d in range(1, 30):
            A = [a, a + d, a + 2 * d]
            if A[2] > 128:
                break
            for b in range(1, 80):
                for e in range(1, 30):
                    B = [b, b + e, b + 2 * e]
                    if B[2] > 128:
                        break
                    try_pair(A, B)
                if time.time() > deadline:
                    return best_result

    if time.time() > deadline:
        return best_result

    # ---- Strategy 3 (from current): weighted random search ----
    random.seed(42)

    def weight_number(n):
        if n < 2:
            return 1
        score = 1.0
        for p in [2, 3, 5, 7, 11, 13]:
            if n % p == 0:
                score *= 0.3
        if is_prime[n]:
            score *= 2.0
        return score

    candidates = list(range(1, 129))
    weights = [weight_number(n) for n in candidates]

    def weighted_sample(k):
        chosen = set()
        attempts = 0
        while len(chosen) < k and attempts < 1000:
            total_w = sum(w for i, w in enumerate(weights) if i + 1 not in chosen)
            if total_w <= 0:
                break
            r = random.random() * total_w
            cum = 0
            for i, w in enumerate(weights):
                if i + 1 in chosen:
                    continue
                cum += w
                if cum >= r:
                    chosen.add(i + 1)
                    break
            attempts += 1
        if len(chosen) < k:
            return None
        return sorted(list(chosen))

    for _ in range(150000):
        if time.time() > deadline:
            return best_result
        A = weighted_sample(3)
        B = weighted_sample(3)
        if A is None or B is None:
            continue
        try_pair(A, B)

    if time.time() > deadline:
        return best_result

    # ---- Strategy 4 (from current): local search around best ----
    if best_result:
        A_best = best_result[0][:]
        B_best = best_result[1][:]
        for _ in range(50000):
            if time.time() > deadline:
                return best_result
            A_new = A_best[:]
            idx = random.randint(0, 2)
            delta = random.randint(-5, 5)
            A_new[idx] = max(1, min(128, A_new[idx] + delta))
            if len(set(A_new)) == 3:
                try_pair(A_new, B_best)
            B_new = B_best[:]
            idx = random.randint(0, 2)
            delta = random.randint(-5, 5)
            B_new[idx] = max(1, min(128, B_new[idx] + delta))
            if len(set(B_new)) == 3:
                try_pair(A_best, B_new)

    if time.time() > deadline:
        return best_result

    # ---- Strategy 5 (from current): distinct-gap random sets ----
    def gen_distinct_gap_set():
        a0 = random.randint(1, 50)
        d1 = random.randint(10, 40)
        d2 = random.randint(10, 40)
        while d2 == d1:
            d2 = random.randint(10, 40)
        A = [a0, a0 + d1, a0 + d1 + d2]
        if A[2] > 128:
            return None
        if len(set(A)) != 3:
            return None
        return sorted(A)

    for _ in range(80000):
        if time.time() > deadline:
            return best_result
        A = gen_distinct_gap_set()
        B = gen_distinct_gap_set()
        if A is None or B is None:
            continue
        try_pair(A, B)

    return best_result
