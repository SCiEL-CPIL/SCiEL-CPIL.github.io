"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute GCD for numbers up to 256
    MAX_SUM = 256
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_pairwise_coprime(nums):
        """Check if all distinct values in nums are pairwise coprime."""
        distinct = list(set(nums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(distinct[i], distinct[j]) > 1:
                    return False
        return True
    
    # Precompute primes up to 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    # Known valid solution with cardinality 9 (all sums are distinct primes)
    # A = [1, 2, 4], B = [3, 7, 11]
    # Sums: 4,8,12,5,9,13,7,11,15 -> not all prime
    # Let's use a verified construction:
    # A = [1, 3, 5], B = [2, 6, 10]
    # Sums: 3,7,11, 5,9,13, 7,11,15 -> distinct: {3,5,7,9,11,13,15} -> 7, not 9
    # 
    # Try: A = [1, 2, 3], B = [4, 8, 16]
    # Sums: 5,9,17, 6,10,18, 7,11,19 -> distinct: {5,6,7,9,10,11,17,18,19} -> 9 distinct
    # Check pairwise coprime: 6 and 9 share 3 -> FAIL
    #
    # A = [1, 2, 4], B = [3, 5, 7]
    # Sums: 4,6,8, 5,7,9, 7,9,11 -> distinct: {4,5,6,7,8,9,11} -> 7
    #
    # Let's do a quick targeted search for a known good solution
    # A = [1, 2, 3], B = [4, 7, 11]
    # Sums: 5,8,12, 6,9,13, 7,10,14 -> distinct: {5,6,7,8,9,10,12,13,14} -> 9
    # 6&8 share 2 -> FAIL
    #
    # Use the search to find one, but with a faster approach
    best_result = [[1, 7, 13], [4, 10, 16]]
    best_card = 5
    
    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not is_pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    # Fast path: try some hand-picked candidates first
    candidates = [
        ([1, 2, 4], [3, 5, 7]),
        ([1, 2, 3], [4, 8, 16]),
        ([1, 3, 5], [2, 6, 10]),
        ([1, 2, 4], [5, 7, 11]),
        ([1, 3, 7], [2, 4, 8]),
        ([1, 2, 5], [3, 7, 11]),
        ([1, 3, 4], [2, 5, 8]),
        ([1, 2, 6], [3, 5, 7]),
        ([1, 4, 7], [2, 3, 5]),
        ([1, 2, 9], [3, 5, 7]),
    ]
    
    for A, B in candidates:
        card = evaluate(A, B)
        if card > best_card:
            best_card = card
            best_result = [A, B]
        if card == 9:
            return [A, B]
    
    # Systematic search with pruning
    range_limit = 30
    
    for a1 in range(1, range_limit):
        if time.time() - t_start > TIME_LIMIT - 2:
            break
        for a2 in range(a1 + 1, range_limit):
            if time.time() - t_start > TIME_LIMIT - 2:
                break
            for a3 in range(a2 + 1, range_limit):
                if time.time() - t_start > TIME_LIMIT - 2:
                    break
                A = [a1, a2, a3]
                
                for b1 in range(1, range_limit):
                    if time.time() - t_start > TIME_LIMIT - 2:
                        break
                    for b2 in range(b1 + 1, range_limit):
                        if time.time() - t_start > TIME_LIMIT - 2:
                            break
                        for b3 in range(b2 + 1, range_limit):
                            if time.time() - t_start > TIME_LIMIT - 2:
                                break
                            B = [b1, b2, b3]
                            
                            sums = [a + b for a in A for b in B]
                            distinct_sums = set(sums)
                            
                            # Quick cardinality check
                            if len(distinct_sums) < best_card:
                                continue
                            
                            # Check if all distinct sums are pairwise coprime
                            if is_pairwise_coprime(distinct_sums):
                                card = len(distinct_sums)
                                if card > best_card:
                                    best_card = card
                                    best_result = [A, B]
                                if card == 9:
                                    return [A, B]
    
    return best_result
# EVOLVE-BLOCK-END
