import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Sieve for primes up to 256
    max_sum = 256
    sieve = [True] * (max_sum + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_sum+1, i):
                sieve[j] = False
    prime_set = set(i for i in range(2, max_sum+1) if sieve[i])
    
    # Precompute valid B for each A
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = set()
        for b in range(1, 129):
            if (a + b) in prime_set:
                valid_b_for_a[a].add(b)
    
    # Strategy 1: Parity-based search (all odd primes)
    # For all sums to be odd primes, one set must be all odd, other all even
    odds = [a for a in range(1, 129) if a % 2 == 1]
    evens = [b for b in range(1, 129) if b % 2 == 0]
    
    # Precompute for parity-constrained search
    # For each odd a, find even b such that a+b is prime
    valid_even_b_for_odd_a = {}
    for a in odds:
        valid_even_b_for_odd_a[a] = set()
        for b in evens:
            if (a + b) in prime_set:
                valid_even_b_for_odd_a[a].add(b)
    
    # Search: A = 3 odds, B = 3 evens
    for a1 in odds:
        if time.time() > deadline:
            break
        vb1 = valid_even_b_for_odd_a[a1]
        if len(vb1) < 3:
            continue
        for a2 in odds:
            if a2 <= a1:
                continue
            if time.time() > deadline:
                break
            vb12 = vb1 & valid_even_b_for_odd_a[a2]
            if len(vb12) < 3:
                continue
            for a3 in odds:
                if a3 <= a2:
                    continue
                if time.time() > deadline:
                    break
                valid_b = vb12 & valid_even_b_for_odd_a[a3]
                if len(valid_b) < 3:
                    continue
                b_list = sorted(valid_b)
                # Try combinations of 3 B values
                for B_combo in combinations(b_list, 3):
                    B = list(B_combo)
                    sums = [a + b for a in (a1, a2, a3) for b in B]
                    if len(set(sums)) == 9:
                        # Verify all are pairwise coprime
                        s_set = sorted(set(sums))
                        valid = True
                        for i in range(len(s_set)):
                            for j in range(i + 1, len(s_set)):
                                if math.gcd(s_set[i], s_set[j]) != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            return [[a1, a2, a3], B]
    
    # Strategy 2: General search (allowing any primes, checking pairwise coprime)
    for a1 in range(1, 129):
        if time.time() > deadline:
            break
        vb1 = valid_b_for_a[a1]
        for a2 in range(a1 + 1, 129):
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
                for B_combo in combinations(b_list, 3):
                    B = list(B_combo)
                    sums = [a + b for a in (a1, a2, a3) for b in B]
                    s_set = sorted(set(sums))
                    if len(s_set) < 9:
                        continue
                    # Check pairwise coprime
                    valid = True
                    for i in range(len(s_set)):
                        for j in range(i + 1, len(s_set)):
                            if math.gcd(s_set[i], s_set[j]) != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(s_set) == 9:
                        return [[a1, a2, a3], B]
    
    # Fallback: try to find any valid configuration
    # Use known good structure
    for a1 in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]:
        for a2 in [a1 + 2, a1 + 4, a1 + 6, a1 + 8, a1 + 10]:
            if a2 > 128:
                continue
            for a3 in [a2 + 2, a2 + 4, a2 + 6, a2 + 8, a2 + 10]:
                if a3 > 128:
                    continue
                A = [a1, a2, a3]
                # Find B values that give distinct sums
                valid_b = set()
                for b in range(1, 129):
                    sums = [a + b for a in A]
                    if all(s in prime_set for s in sums):
                        # Check pairwise coprime with existing sums
                        existing_sums = set()
                        for b2 in valid_b:
                            existing_sums.update(a + b2 for a in A)
                        new_sums = set(sums)
                        all_sums = existing_sums | new_sums
                        pairwise_ok = True
                        for i, s1 in enumerate(all_sums):
                            for s2 in list(all_sums)[i+1:]:
                                if math.gcd(s1, s2) != 1:
                                    pairwise_ok = False
                                    break
                            if not pairwise_ok:
                                break
                        if pairwise_ok:
                            valid_b.add(b)
                if len(valid_b) >= 3:
                    b_list = sorted(valid_b)
                    for B_combo in combinations(b_list, 3):
                        B = list(B_combo)
                        sums = [a + b for a in A for b in B]
                        s_set = set(sums)
                        if len(s_set) == 9:
                            # Final check
                            valid = True
                            for i in range(len(s_set)):
                                for j in range(i + 1, len(s_set)):
                                    if math.gcd(list(s_set)[i], list(s_set)[j]) != 1:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                return [A, B]
    
    return [[1, 7, 13], [4, 10, 16]]
