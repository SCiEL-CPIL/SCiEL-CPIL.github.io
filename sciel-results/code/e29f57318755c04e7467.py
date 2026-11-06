"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill

    # Precompute sieve for primality up to 256 (max sum = 128+128)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False

    def is_pairwise_coprime(vals):
        """Check if all distinct values in vals are pairwise coprime."""
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    def count_distinct_coprime(A, B):
        """Count distinct values in A+B that are pairwise coprime. Return 0 if not all are pairwise coprime."""
        sums = set(a + b for a in A for b in B)
        if is_pairwise_coprime(sums):
            return len(sums)
        return 0

    # Best known result from seed: 5 distinct coprime sums
    best_score = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Strategy: Try to find 9 distinct primes as sums (maximum possible for 3x3)
    # If we can find A and B such that all 9 sums are distinct primes, that's optimal.
    
    # First, try a focused search: pick 3 primes as targets for sums,
    # then see if we can decompose them into A+B form.
    
    # Alternative: Brute force over small ranges first, then expand.
    # Since we need 3 distinct integers in [1,128], let's search systematically.
    
    # Key insight: If all 9 sums are primes, they're automatically pairwise coprime.
    # So we want A+B to produce 9 distinct primes.
    
    # Let's try a smarter search: iterate over possible A values and for each,
    # try B values, checking if all 9 sums are distinct primes.
    
    # To make this efficient, let's first identify candidate sets.
    # A sum a+b is prime. For fixed A = [a1, a2, a3], we need B = [b1, b2, b3]
    # such that all a_i + b_j are distinct primes.
    
    # Let's try a more targeted approach:
    # 1. Pick a1, a2, a3 from a reasonable range
    # 2. For each combination, find b values that make all sums prime
    
    # But this is still 128^3 * 128^3 which is too large.
    
    # Better: Use the fact that if a+b is prime, and a is fixed, b must be such that a+b is prime.
    # For a given a, the valid b values are those where a+b is prime and in [1,256].
    
    # Let's precompute for each a in [1,128], the set of b in [1,128] where a+b is prime.
    prime_b_for_a = {}
    for a in range(1, 129):
        prime_b_for_a[a] = set()
        for b in range(1, 129):
            if is_prime[a + b]:
                prime_b_for_a[a].add(b)
    
    # Now search: for each triple (a1, a2, a3), find b1, b2, b3 such that
    # all 9 sums a_i + b_j are distinct primes.
    # We can iterate over a1, a2, a3 and for each, find common b values that work.
    
    # For a given (a1, a2, a3), b must be in prime_b_for_a[a1] ∩ prime_b_for_a[a2] ∩ prime_b_for_a[a3]
    # for all sums to be prime. But we also need them to be distinct.
    
    # Let's iterate over triples of a values:
    best_found = False
    
    # Limit search to reasonable ranges to stay within time
    # Try a values in [1, 50] first, then expand if needed
    search_range = range(1, 51)
    
    for a1 in search_range:
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2 in search_range:
            if a2 <= a1:
                continue
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3 in search_range:
                if a3 <= a2:
                    continue
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
                
                # Find b values that make all sums with a1, a2, a3 prime
                common_b = prime_b_for_a[a1] & prime_b_for_a[a2] & prime_b_for_a[a3]
                
                if len(common_b) < 3:
                    continue
                
                # Now try all triples of b from common_b
                b_list = sorted(common_b)
                if len(b_list) < 3:
                    continue
                
                for b1 in b_list:
                    for b2 in b_list:
                        if b2 <= b1:
                            continue
                        for b3 in b_list:
                            if b3 <= b2:
                                continue
                            
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            
                            # Check all 9 sums are distinct primes
                            sums = set()
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if s in sums or not is_prime[s]:
                                        valid = False
                                        break
                                    sums.add(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                best_score = 9
                                best_A = A
                                best_B = B
                                best_found = True
                                return [best_A, best_B]
    
    # If we didn't find 9, try to find the best we can with a broader but still bounded search
    # Fall back to a more general search for pairwise coprime (not necessarily prime) sums
    
    # Try a moderate brute force over smaller ranges
    range2 = range(1, 30)
    for A in itertools.combinations(range2, 3):
        if time.time() - t_start > TIME_LIMIT - 3:
            break
        for B in itertools.combinations(range2, 3):
            score = count_distinct_coprime(list(A), list(B))
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
                if best_score == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
