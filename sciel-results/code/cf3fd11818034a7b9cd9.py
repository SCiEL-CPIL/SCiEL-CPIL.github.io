"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def are_pairwise_coprime(nums):
    unique = list(set(nums))
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if gcd(unique[i], unique[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Default fallback: 5 distinct pairwise-coprime sums
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # --- Sieve primes up to 256 ---
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, n + 1, i):
                    is_p[j] = False
        return is_p

    is_prime = sieve(256)

    # --- Modular pre-filter for A triples ---
    # For a_i + b to be prime for all i, the differences a_i - a_1 must not
    # be divisible by any small prime p (unless a_1 ≡ 0 mod p, but then
    # a_1 + b ≡ b mod p must be 0, forcing b ≡ 0 mod p, which is very
    # restrictive). So we require gcd(a_i - a_1, p) == 1 for all small p.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    def a_triple_viable(a1, a2, a3):
        for p in small_primes:
            d2 = (a2 - a1) % p
            d3 = (a3 - a1) % p
            if d2 == 0 or d3 == 0:
                # a2+b ≡ a1+b (mod p) and a3+b ≡ a1+b (mod p)
                # This means if a1+b ≡ 0 mod p, then all three are 0 mod p
                # (bad for p > 256, but for p <= 23, a1+b is at most 256,
                #  so a1+b could be p, 2p, 3p, ... which are composite)
                # Actually for p=2: a2-a1 even means a2+b and a1+b have same
                # parity. If both odd, fine. If both even, one is 2 (ok) or
                # composite. Let's be conservative: skip if d==0 for p>2.
                if p > 2:
                    return False
        return True

    # --- Phase 1: Search for 9 distinct prime sums ---
    # For each viable A triple, find all b in [1,128] where all a_i+b are prime.
    # Then pick 3 b's giving 9 distinct sums.
    found_9 = False

    for a1 in range(1, 129):
        if time.time() > deadline - 3:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline - 3:
                break
            if not a_triple_viable(a1, a2, 0):  # quick check for a2
                # Check pairwise for a1,a2
                for p in small_primes:
                    if p > 2 and (a2 - a1) % p == 0:
                        break
                else:
                    pass  # continue
                # Actually let's just check the full triple later; skip this
                # micro-optimization and do it properly in the inner loop.
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline - 3:
                    break

                # Check modular viability
                viable = True
                for p in small_primes:
                    if p > 2:
                        if (a2 - a1) % p == 0 or (a3 - a1) % p == 0 or (a3 - a2) % p == 0:
                            viable = False
                            break
                if not viable:
                    continue

                # Find all valid b
                valid_b = []
                for b in range(1, 129):
                    if is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]:
                        valid_b.append(b)

                if len(valid_b) < 3:
                    continue

                # Try to find 3 b's giving 9 distinct sums
                # Collision: a_i + b_j = a_k + b_l iff a_i - a_k = b_l - b_j
                # For 9 distinct, we need no such collision.
                # Since valid_b is usually small, try all triples.
                # But first, if len(valid_b) is large, sample.
                if len(valid_b) <= 25:
                    candidates = valid_b
                else:
                    candidates = valid_b[:25]

                for b1, b2, b3 in combinations(candidates, 3):
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    sums = [A[i] + B[j] for i in range(3) for j in range(3)]
                    if len(set(sums)) == 9:
                        best_A = A
                        best_B = B
                        best_count = 9
                        found_9 = True
                        break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break

    if found_9:
        return [best_A, best_B]

    # --- Phase 2: Broader pairwise-coprime search ---
    # Try to find A, B where all 9 sums are distinct and pairwise coprime.
    # Use a targeted approach: pick A with small values, then search B.
    import random
    rng = random.Random(42)

    # Structured search: try A triples from a curated set
    # Use A values that are pairwise coprime-friendly (small, diverse)
    structured_A = []
    for a1 in range(1, 40):
        for a2 in range(a1 + 1, 40):
            for a3 in range(a2 + 1, 40):
                # Quick filter: a_i should not share small factors
                # to make sums more likely coprime
                structured_A.append([a1, a2, a3])

    for A in structured_A:
        if time.time() > deadline - 2:
            break
        # Find B values where all sums are pairwise coprime
        # For each candidate b, check if a_i + b is "good" (prime or prime power)
        # Then pick 3 b's
        valid_b = []
        for b in range(1, 129):
            sums_for_b = [A[i] + b for i in range(3)]
            # Check if these are all prime or at least pairwise coprime with each other
            # and will be coprime with sums from other b's
            valid_b.append(b)

        # Try random subsets of valid_b
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            B = sorted(rng.sample(range(1, 129), 3))
            sums = [A[i] + B[j] for i in range(3) for j in range(3)]
            distinct_sums = set(sums)
            count = len(distinct_sums)
            if count > best_count and are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]

    # --- Phase 3: Random search ---
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        sums = [A[i] + B[j] for i in range(3) for j in range(3)]
        distinct_sums = set(sums)
        count = len(distinct_sums)
        if count > best_count:
            if are_pairwise_coprime(distinct_sums):
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
        attempts += 1

    return [best_A, best_B]
# EVOLVE-BLOCK-END
