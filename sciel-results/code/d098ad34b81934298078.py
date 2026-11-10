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
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return is_p

    is_prime = sieve(256)

    # --- Phase 1: Search for 9 distinct prime sums ---
    # For each A triple, find all b in [1,128] where all a_i+b are prime.
    # Then pick 3 b's giving 9 distinct sums.
    
    # Precompute for each a the set of b values that make a+b prime
    # b_valid[a] = set of b in [1,128] such that a+b is prime
    b_valid = [set() for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            if is_prime[a + b]:
                b_valid[a].add(b)

    found_9 = False

    for a1 in range(1, 129):
        if time.time() > deadline - 3:
            break
        for a2 in range(a1 + 1, 129):
            if time.time() > deadline - 3:
                break
            # Intersection of valid b's for a1 and a2
            common_b12 = b_valid[a1] & b_valid[a2]
            if len(common_b12) < 3:
                continue
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline - 3:
                    break
                # Intersection of valid b's for all three
                valid_b = common_b12 & b_valid[a3]
                if len(valid_b) < 3:
                    continue
                
                valid_b_list = sorted(valid_b)
                
                # Try to find 3 b's giving 9 distinct sums
                # Collision: a_i + b_j = a_k + b_l iff a_i - a_k = b_l - b_j
                # Since a1 < a2 < a3, differences are a2-a1, a3-a1, a3-a2
                # We need to avoid b_j - b_l being any of these differences
                diffs = {a2 - a1, a3 - a1, a3 - a2}
                
                candidates = valid_b_list if len(valid_b_list) <= 30 else valid_b_list[:30]
                
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
    # Try to find A, B where all 9 sums are distinct and pairwise coprime
    # (not necessarily all prime)
    import random
    rng = random.Random(42)

    # Structured search with small A values
    for a1 in range(1, 30):
        if time.time() > deadline - 2:
            break
        for a2 in range(a1 + 1, 30):
            if time.time() > deadline - 2:
                break
            for a3 in range(a2 + 1, 30):
                if time.time() > deadline - 2:
                    break
                A = [a1, a2, a3]
                # Find b values where sums are "good" (prime or prime power)
                valid_b = []
                for b in range(1, 129):
                    sums_for_b = [A[i] + b for i in range(3)]
                    # Check if all are prime or at least coprime-friendly
                    valid_b.append(b)
                
                # Try random subsets
                for _ in range(100):
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
