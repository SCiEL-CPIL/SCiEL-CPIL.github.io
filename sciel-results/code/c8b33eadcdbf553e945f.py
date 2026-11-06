"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    best_score = 5
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Precompute smallest prime factor for fast factorization
    max_val = 256
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        """Return set of prime factors of n using spf table."""
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def is_pairwise_coprime(vals):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    # Use odd for A, even for B to avoid all-even sums
    odd_vals = list(range(1, 129, 2))
    even_vals = list(range(2, 129, 2))
    
    # Phase 1: Standard odd/even search
    A_search = odd_vals[:60]   # 1 to 119
    B_search = even_vals[:60]  # 2 to 120
    
    # Precompute gcd table for faster checks
    max_sum = 128 + 128
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(1, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def is_pairwise_coprime_fast(vals, gt):
        n = len(vals)
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                if gt[vi][vals[j]] != 1:
                    return False
        return True
    
    def check_with_forbidden_primes(sums, forbidden):
        """Check if sums are pairwise coprime using forbidden prime set."""
        for s in sums:
            for p in prime_factors(s):
                if p in forbidden:
                    return False
            forbidden.update(prime_factors(s))
        return True
    
    for A in itertools.combinations(A_search, 3):
        if time.time() > deadline:
            return best_result
        a1, a2, a3 = A
        
        valid_b = []
        for b in B_search:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            if (s1 > max_sum or s2 > max_sum or s3 > max_sum):
                continue
            # Check that sums within this B are pairwise coprime
            if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                valid_b.append((b, s1, s2, s3))
        
        if len(valid_b) < 3:
            continue
        
        # Limit to top candidates for performance
        if len(valid_b) > 40:
            valid_b = valid_b[:40]
        
        n = len(valid_b)
        for i in range(n):
            if time.time() > deadline:
                return best_result
            b1, s1_1, s2_1, s3_1 = valid_b[i]
            sums1 = [s1_1, s2_1, s3_1]
            
            # Build forbidden primes from first B
            forbidden1 = set()
            for s in sums1:
                forbidden1.update(prime_factors(s))
            
            for j in range(i + 1, n):
                b2, s1_2, s2_2, s3_2 = valid_b[j]
                sums12 = sums1 + [s1_2, s2_2, s3_2]
                
                # Quick check with forbidden primes
                if not check_with_forbidden_primes([s1_2, s2_2, s3_2], set(forbidden1)):
                    continue
                
                if not is_pairwise_coprime_fast(sums12, gcd_table):
                    continue
                
                forbidden12 = set(forbidden1)
                for s in [s1_2, s2_2, s3_2]:
                    forbidden12.update(prime_factors(s))
                
                for k in range(j + 1, n):
                    b3, s1_3, s2_3, s3_3 = valid_b[k]
                    all_sums = sums12 + [s1_3, s2_3, s3_3]
                    if len(set(all_sums)) <= best_score:
                        continue
                    
                    # Quick check with forbidden primes
                    if not check_with_forbidden_primes([s1_3, s2_3, s3_3], set(forbidden12)):
                        continue
                    
                    if is_pairwise_coprime_fast(all_sums, gcd_table):
                        distinct_count = len(set(all_sums))
                        if distinct_count > best_score:
                            best_score = distinct_count
                            best_result = [list(A), [b1, b2, b3]]
                            if best_score == 9:
                                return best_result
    
    # Phase 2: Try mixed parity - some odd in B, some even in A
    if time.time() < deadline - 20:
        # Try A with mixed values and B with mixed values
        A_search2 = list(range(1, 80))  # More diverse
        B_search2 = list(range(2, 80))
        
        # Pre-filter: avoid obvious common factors
        for A in itertools.combinations(A_search2, 3):
            if time.time() > deadline - 10:
                break
            a1, a2, a3 = A
            # Skip if all same parity
            parities = [a % 2 for a in (a1, a2, a3)]
            if len(set(parities)) == 1:
                continue
            
            valid_b = []
            for b in B_search2:
                s1 = a1 + b
                s2 = a2 + b
                s3 = a3 + b
                if (s1 > max_sum or s2 > max_sum or s3 > max_sum):
                    continue
                if gcd_table[s1][s2] == 1 and gcd_table[s1][s3] == 1 and gcd_table[s2][s3] == 1:
                    valid_b.append((b, s1, s2, s3))
            
            if len(valid_b) < 3:
                continue
            
            if len(valid_b) > 30:
                valid_b = valid_b[:30]
            
            n = len(valid_b)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                b1, s1_1, s2_1, s3_1 = valid_b[i]
                sums1 = [s1_1, s2_1, s3_1]
                forbidden1 = set()
                for s in sums1:
                    forbidden1.update(prime_factors(s))
                
                for j in range(i + 1, n):
                    b2, s1_2, s2_2, s3_2 = valid_b[j]
                    if not check_with_forbidden_primes([s1_2, s2_2, s3_2], set(forbidden1)):
                        continue
                    sums12 = sums1 + [s1_2, s2_2, s3_2]
                    if not is_pairwise_coprime_fast(sums12, gcd_table):
                        continue
                    forbidden12 = set(forbidden1)
                    for s in [s1_2, s2_2, s3_2]:
                        forbidden12.update(prime_factors(s))
                    
                    for k in range(j + 1, n):
                        b3, s1_3, s2_3, s3_3 = valid_b[k]
                        all_sums = sums12 + [s1_3, s2_3, s3_3]
                        if len(set(all_sums)) <= best_score:
                            continue
                        if not check_with_forbidden_primes([s1_3, s2_3, s3_3], set(forbidden12)):
                            continue
                        if is_pairwise_coprime_fast(all_sums, gcd_table):
                            distinct_count = len(set(all_sums))
                            if distinct_count > best_score:
                                best_score = distinct_count
                                best_result = [list(A), [b1, b2, b3]]
                                if best_score == 9:
                                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
