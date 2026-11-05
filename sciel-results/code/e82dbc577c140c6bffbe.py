"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80

    # Sieve for primes up to 256
    sieve = [True] * 257
    sieve[0] = sieve[1] = False
    for i in range(2, 16):
        if sieve[i]:
            for j in range(i*i, 257, i):
                sieve[j] = False
    prime_set = set(i for i in range(2, 257) if sieve[i])

    def pairwise_coprime(vals):
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    # Precompute valid_b_for_a: for each a, which b values make a+b prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = set(b for b in range(1, 129) if (a + b) in prime_set)

    best_count = 0
    best_A = None
    best_B = None

    # Search for A triples where the intersection of valid_b_for_a is large
    # Then try to find B triples from that intersection giving 9 distinct prime sums

    # Iterate over A triples in a structured way
    # Key: for 9 distinct sums, we need a_i + b_j all different
    # This means differences between a's must not equal differences between b's (in absolute value)
    
    for a1 in range(1, 65):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 97):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                A = (a1, a2, a3)
                valid_b = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]
                if len(valid_b) < 3:
                    continue
                
                b_list = sorted(valid_b)
                # Try combinations of 3 from valid_b
                # To ensure 9 distinct sums, we need:
                # a1+b_i != a1+b_j (always true for distinct b)
                # a1+b_i != a2+b_j means b_j - b_i != a1 - a2
                # So we need b_j - b_i not in {a1-a2, a1-a3, a2-a3} and their negatives
                
                diffs_a = set([a1-a2, a1-a3, a2-a3, a2-a1, a3-a1, a3-a2])
                
                # Try to find 3 b values where no pair difference is in diffs_a
                found = False
                for i in range(len(b_list)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, len(b_list)):
                        if time.time() > deadline:
                            break
                        dj = b_list[j] - b_list[i]
                        if dj in diffs_a:
                            continue
                        for k in range(j+1, len(b_list)):
                            dk = b_list[k] - b_list[j]
                            if dk in diffs_a:
                                continue
                            dk2 = b_list[k] - b_list[i]
                            if dk2 in diffs_a:
                                continue
                            B = (b_list[i], b_list[j], b_list[k])
                            sums = [a + b for a in A for b in B]
                            if len(set(sums)) == 9 and all(s in prime_set for s in sums):
                                best_A = list(A)
                                best_B = list(B)
                                best_count = 9
                                found = True
                                break
                            if time.time() > deadline:
                                break
                        if found or time.time() > deadline:
                            break
                    if found or time.time() > deadline:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break

    if best_count == 9:
        return [best_A, best_B]

    # If 9 not found, fall back to finding best pairwise coprime set
    # Try more combinations
    best_count = 0
    best_A = None
    best_B = None

    for a1 in range(1, 51):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 80):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                A = (a1, a2, a3)
                valid_b = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]
                if len(valid_b) < 3:
                    continue
                b_list = sorted(valid_b)
                for i in range(min(5, len(b_list) - 2)):
                    B = (b_list[i], b_list[i+1], b_list[i+2])
                    sums = [a + b for a in A for b in B]
                    distinct_sums = set(sums)
                    if pairwise_coprime(distinct_sums) and len(distinct_sums) > best_count:
                        best_count = len(distinct_sums)
                        best_A = list(A)
                        best_B = list(B)
                        if best_count >= 9:
                            return [best_A, best_B]

    if best_A is not None:
        return [best_A, best_B]

    # Fallback
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
