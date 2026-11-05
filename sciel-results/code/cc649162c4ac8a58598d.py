"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    def pairwise_coprime(vals):
        """Check if all distinct values are pairwise coprime."""
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Compute the number of distinct sums that are pairwise coprime."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_pair(best_A, best_B)
    
    # Strategy: Search for A, B where all 9 sums are distinct primes
    # This guarantees pairwise coprimality and max cardinality of 9
    
    # Precompute primes up to 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 16):
        if is_prime[i]:
            for j in range(i * i, 257, i):
                is_prime[j] = False
    
    # For efficient search, precompute for each prime p, which pairs (a,b) sum to p
    # But that's complex. Instead, let's do a structured search.
    
    # Key insight: if all a+b are primes, they're pairwise coprime.
    # We need a_i + b_j to be prime for all i,j, and ideally all 9 distinct.
    
    # Let's search over small values first, then expand.
    # Use a greedy/local search approach.
    
    # Phase 1: Try all combinations of small numbers
    # Limit to values 1-40 for initial search to keep it fast
    max_val_phase1 = 40
    candidates = list(range(1, max_val_phase1 + 1))
    
    # For each A, find best B
    # Optimization: for a given A, check B candidates
    # A set A is "good" if there exists B such that all a+b are prime
    
    # Let's iterate over A sets (combinations of 3 from 1..40)
    # That's C(40,3) = 9880 A sets
    # For each, iterate over B sets: another 9880
    # Total: ~97M pairs - too many for brute force in 80s
    
    # Better: for each A, find valid B sets
    # A set A is valid with B if for all a in A, b in B: a+b is prime
    # So for a given A, B must be a set of 3 values where each b satisfies:
    #   for all a in A: a+b is prime
    
    # Let's precompute for each b in [1,40], which A values make a+b prime
    # Actually, let's think differently.
    
    # For a fixed A = {a1, a2, a3}, a value b is "compatible" if:
    #   a1+b, a2+b, a3+b are all prime
    # Then B must be 3 distinct compatible values.
    
    # Let's iterate over A sets and count compatible B values.
    
    # Precompute prime lookup for sums up to 80 (40+40)
    is_prime_small = [False] * 81
    for p in range(2, 81):
        is_prime_small[p] = is_prime[p]
    
    # For each b in [1, 40], precompute which a values make a+b prime
    # Actually, let's just iterate.
    
    # For each A (combination of 3 from 1..40):
    #   Find all b in [1,40] such that a1+b, a2+b, a3+b are all prime
    #   If at least 3 such b exist, pick any 3 and evaluate
    
    # This is C(40,3) = 9880 iterations, each O(40) = ~400K ops. Fast!
    
    for a1 in range(1, max_val_phase1 + 1):
        for a2 in range(a1 + 1, max_val_phase1 + 1):
            for a3 in range(a2 + 1, max_val_phase1 + 1):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                # Find compatible b values
                compatible_b = []
                for b in range(1, max_val_phase1 + 1):
                    if (is_prime_small[a1 + b] and is_prime_small[a2 + b] and is_prime_small[a3 + b]):
                        compatible_b.append(b)
                if len(compatible_b) >= 3:
                    # Try different B choices to maximize distinct sums
                    # First, just take the first 3
                    B = compatible_b[:3]
                    s = score_pair(A, B)
                    if s > best_score:
                        best_score = s
                        best_A = A[:]
                        best_B = B[:]
                        if best_score == 9:
                            # Maximum possible
                            return [best_A, best_B]
                    # Try a few other combinations of compatible_b
                    if len(compatible_b) > 3:
                        # Try last 3, middle 3, etc.
                        for b_idx in range(1, min(5, len(compatible_b) - 2)):
                            B2 = compatible_b[b_idx:b_idx + 3]
                            s2 = score_pair(A, B2)
                            if s2 > best_score:
                                best_score = s2
                                best_A = A[:]
                                best_B = B2[:]
                                if best_score == 9:
                                    return [best_A, best_B]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Phase 2: Expand to larger values if time permits
    if time.time() < deadline - 5:
        max_val_phase2 = 80
        # For larger range, use a smarter approach
        # Focus on A values that are likely to work
        # Try A values that are odd (so a+b is even only if b is odd, but 2 is the only even prime)
        # Actually, for a+b to be prime > 2, one of a,b must be odd and the other even.
        # Or both odd (sum even, only prime is 2, but min sum is 1+1=2).
        # Wait: if a and b are both odd, a+b is even. The only even prime is 2.
        # So for a+b to be an odd prime, one must be odd and the other even.
        # This is a strong constraint!
        
        # So: if all 9 sums are odd primes, then for each pair (a,b), one is odd and one is even.
        # This means: either all A are odd and all B are even, or all A are even and all B are odd.
        # (Can't mix within a set, because then some a1+a2 would have same parity as a1+b, etc.)
        # Wait, no. The constraint is on a+b for a in A, b in B.
        # If A has both odd and even, and B has both odd and even, then:
        #   odd_a + even_b = odd (could be prime)
        #   even_a + odd_b = odd (could be prime)  
        #   odd_a + odd_b = even (only prime is 2, but sums >= 2)
        #   even_a + even_b = even (only prime is 2)
        # So if A and B both have mixed parity, some sums will be even > 2, not prime.
        # Therefore, for all sums to be odd primes:
        #   Either A all odd, B all even
        #   Or A all even, B all odd
        #   Or one set all odd, other all even
        
        # This dramatically reduces the search space!
        
        # Let's search with this constraint for larger values.
        odd_vals = list(range(1, max_val_phase2 + 1, 2))  # 1,3,5,...,79
        even_vals = list(range(2, max_val_phase2 + 1, 2))  # 2,4,6,...,80
        
        # Case 1: A odd, B even
        # For a in A (odd), b in B (even): a+b is odd. Good.
        # Need a+b to be prime for all pairs.
        
        # For each A (3 from odd_vals), find B (3 from even_vals) such that all sums are prime.
        # C(40,3) * C(40,3) is still too large.
        # But for each A, find compatible B values.
        
        for a1 in odd_vals:
            if time.time() > deadline:
                break
            for a2 in odd_vals:
                if a2 <= a1:
                    continue
                if time.time() > deadline:
                    break
                for a3 in odd_vals:
                    if a3 <= a2:
                        continue
                    if time.time() > deadline:
                        break
                    A = [a1, a2, a3]
                    compatible_b = []
                    for b in even_vals:
                        if (is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]):
                            compatible_b.append(b)
                    if len(compatible_b) >= 3:
                        B = compatible_b[:3]
                        s = score_pair(A, B)
                        if s > best_score:
                            best_score = s
                            best_A = A[:]
                            best_B = B[:]
                            if best_score == 9:
                                return [best_A, best_B]
                        # Try other combinations
                        if len(compatible_b) > 3:
                            for b_idx in range(1, min(5, len(compatible_b) - 2)):
                                B2 = compatible_b[b_idx:b_idx + 3]
                                s2 = score_pair(A, B2)
                                if s2 > best_score:
                                    best_score = s2
                                    best_A = A[:]
                                    best_B = B2[:]
                                    if best_score == 9:
                                        return [best_A, best_B]
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        # Case 2: A even, B odd
        if time.time() < deadline - 3:
            for a1 in even_vals:
                if time.time() > deadline:
                    break
                for a2 in even_vals:
                    if a2 <= a1:
                        continue
                    if time.time() > deadline:
                        break
                    for a3 in even_vals:
                        if a3 <= a2:
                            continue
                        if time.time() > deadline:
                            break
                        A = [a1, a2, a3]
                        compatible_b = []
                        for b in odd_vals:
                            if (is_prime[a1 + b] and is_prime[a2 + b] and is_prime[a3 + b]):
                                compatible_b.append(b)
                        if len(compatible_b) >= 3:
                            B = compatible_b[:3]
                            s = score_pair(A, B)
                            if s > best_score:
                                best_score = s
                                best_A = A[:]
                                best_B = B[:]
                                if best_score == 9:
                                    return [best_A, best_B]
                            if len(compatible_b) > 3:
                                for b_idx in range(1, min(5, len(compatible_b) - 2)):
                                    B2 = compatible_b[b_idx:b_idx + 3]
                                    s2 = score_pair(A, B2)
                                    if s2 > best_score:
                                        best_score = s2
                                        best_A = A[:]
                                        best_B = B2[:]
                                        if best_score == 9:
                                            return [best_A, best_B]
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
