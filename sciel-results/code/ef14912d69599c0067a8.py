"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 70  # Safety margin before 90s kill
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def pairwise_coprime(vals):
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    # Precompute primes up to 256 (max sum = 128+128) using sieve
    max_sum = 256
    sieve = [True] * (max_sum + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_sum+1, i):
                sieve[j] = False
    prime_set = set(i for i in range(2, max_sum+1) if sieve[i])
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Precompute valid_b_for_a: for each a, which b values make a+b prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = set()
        for b in range(1, 129):
            if (a + b) in prime_set:
                valid_b_for_a[a].add(b)
    
    # Phase 1: Search for 9 distinct prime sums (maximum possible)
    # Try all A combinations with a1 < a2 < a3
    # For each A, find B values where all a_i + b are prime
    # Then check if we can get 9 distinct sums
    
    found_9 = False
    a_max = 128
    a1_max = 64
    a2_max = 96
    
    for a1 in range(1, a1_max + 1):
        if time.time() > deadline:
            break
        vb1 = valid_b_for_a[a1]
        for a2 in range(a1 + 1, a2_max + 1):
            if time.time() > deadline:
                break
            vb12 = vb1 & valid_b_for_a[a2]
            if len(vb12) < 3:
                continue
            for a3 in range(a2 + 1, a_max + 1):
                if time.time() > deadline:
                    break
                valid_b = vb12 & valid_b_for_a[a3]
                if len(valid_b) < 3:
                    continue
                b_list = sorted(valid_b)
                # Try different B combinations to maximize distinct sums
                # First try consecutive triples
                for i in range(len(b_list) - 2):
                    B = (b_list[i], b_list[i+1], b_list[i+2])
                    sums = [a + b for a in (a1, a2, a3) for b in B]
                    if len(set(sums)) == 9:
                        best_A = [a1, a2, a3]
                        best_B = list(B)
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
    
    # Phase 2: Search for 8 or 7 distinct pairwise coprime sums
    # Allow non-prime sums as long as they're pairwise coprime
    # For each A, try multiple B combinations
    
    a1_max2 = 80
    a2_max2 = 110
    
    for a1 in range(1, a1_max2 + 1):
        if time.time() > deadline:
            break
        vb1 = valid_b_for_a[a1]
        for a2 in range(a1 + 1, a2_max2 + 1):
            if time.time() > deadline:
                break
            vb12 = vb1 & valid_b_for_a[a2]
            if len(vb12) < 3:
                continue
            for a3 in range(a2 + 1, 129):
                if time.time() > deadline:
                    break
                valid_b = vb12 & valid_b_for_a[a3]
                if len(valid_b) < 3:
                    continue
                b_list = sorted(valid_b)
                # Try several B combinations
                for i in range(min(5, len(b_list) - 2)):
                    B = (b_list[i], b_list[i+1], b_list[i+2])
                    sums = [a + b for a in (a1, a2, a3) for b in B]
                    distinct_sums = set(sums)
                    if len(distinct_sums) > best_count and pairwise_coprime(distinct_sums):
                        best_count = len(distinct_sums)
                        best_A = [a1, a2, a3]
                        best_B = list(B)
                        if best_count >= 9:
                            return [best_A, best_B]
    
    if best_A is not None:
        return [best_A, best_B]
    
    # Fallback: use the seed solution
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
