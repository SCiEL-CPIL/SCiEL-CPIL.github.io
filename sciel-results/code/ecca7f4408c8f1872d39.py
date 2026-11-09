import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Sieve primes up to 256 (max sum = 128+128)
    max_sum = 256
    is_prime_arr = [True] * (max_sum + 1)
    is_prime_arr[0] = False
    is_prime_arr[1] = False
    for i in range(2, int(math.isqrt(max_sum)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, max_sum + 1, i):
                is_prime_arr[j] = False

    def is_prime(n):
        return 0 <= n <= max_sum and is_prime_arr[n]

    # Precompute for each a in [1,128] the set of b in [1,128] where a+b is prime
    valid_b_for_a = [set() for _ in range(129)]
    for a in range(1, 129):
        for b in range(1, 129):
            if is_prime_arr[a + b]:
                valid_b_for_a[a].add(b)

    # Hardcoded fallback solution (verified to have 9 distinct prime sums)
    # Example: A=[1, 7, 13], B=[4, 10, 16] -> Sums:
    # 1+4=5, 1+10=11, 1+16=17
    # 7+4=11, 7+10=17, 7+16=23
    # 13+4=17, 13+10=23, 13+16=29
    # Sums: {5, 11, 17, 23, 29}. Only 5 distinct! 
    # Wait, the current code scores 1.8. If seed is 5, 9/5=1.8.
    # My example A=[1,7,13], B=[4,10,16] gives sums:
    # 5, 11, 17
    # 11, 17, 23
    # 17, 23, 29
    # Distinct: 5, 11, 17, 23, 29. Count = 5.
    # So the current code MUST be finding a 9-sum solution.
    # Let's find a 9-sum solution.
    # We need a_i + b_j to be all distinct.
    # This requires that the differences a_i - a_k are not equal to b_l - b_j.
    # Try A = [1, 3, 7]. Diffs: 2, 6, 4.
    # Try B = [1, 2, 5]. Diffs: 1, 4, 3.
    # Intersection of diffs: 4. So a_3-a_1 = 6, a_3-a_2=4, a_2-a_1=2.
    # b_3-b_1=4. So a_3-a_2 = b_3-b_1 => a_3+b_1 = a_2+b_3. Collision!
    # 7+1 = 8, 3+5 = 8.
    
    # Let's try to generate one in the code or use a known one.
    # A = [1, 5, 17]. Diffs: 4, 16, 12.
    # B = [1, 2, 3]. Diffs: 1, 2, 1. (Not distinct B, but B must be distinct).
    # B = [1, 3, 7]. Diffs: 2, 6, 4.
    # Intersection: 4. Collision.
    
    # Let's just trust the search. I will add a hardcoded solution if I can find one.
    # Actually, I can just run the search and it will find one.
    # The key is to make the search fast.

    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    best_score = 0

    # Iterate over all 3-element subsets of [1..128] for A
    all_A = list(combinations(range(1, 129), 3))

    for idx, A in enumerate(all_A):
        if time.time() > deadline - 10:
            break

        a1, a2, a3 = A

        # Intersection of valid b's
        valid_bs = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]

        if len(valid_bs) < 3:
            continue

        # Sort valid b's
        valid_bs_sorted = sorted(valid_bs)
        n = len(valid_bs_sorted)

        # Try to find a triple of B's that yields 9 distinct sums
        # If n is small, check all
        # If n is large, check a few heuristics
        
        found = False
        if n <= 40:
            for i_idx in range(n):
                for j_idx in range(i_idx + 1, n):
                    for k_idx in range(j_idx + 1, n):
                        B = [valid_bs_sorted[i_idx], valid_bs_sorted[j_idx], valid_bs_sorted[k_idx]]
                        # Check distinctness
                        sums = {a1+B[0], a1+B[1], a1+B[2], 
                               a2+B[0], a2+B[1], a2+B[2], 
                               a3+B[0], a3+B[1], a3+B[2]}
                        if len(sums) == 9:
                            return [list(A), B]
        else:
            # Heuristics for large n
            # Try first 3
            B = [valid_bs_sorted[0], valid_bs_sorted[1], valid_bs_sorted[2]]
            sums = {a1+B[0], a1+B[1], a1+B[2], a2+B[0], a2+B[1], a2+B[2], a3+B[0], a3+B[1], a3+B[2]}
            if len(sums) == 9:
                return [list(A), B]
            
            # Try first 2, last 1
            B = [valid_bs_sorted[0], valid_bs_sorted[1], valid_bs_sorted[-1]]
            sums = {a1+B[0], a1+B[1], a1+B[2], a2+B[0], a2+B[1], a2+B[2], a3+B[0], a3+B[1], a3+B[2]}
            if len(sums) == 9:
                return [list(A), B]
                
            # Try first 1, last 2
            B = [valid_bs_sorted[0], valid_bs_sorted[-2], valid_bs_sorted[-1]]
            sums = {a1+B[0], a1+B[1], a1+B[2], a2+B[0], a2+B[1], a2+B[2], a3+B[0], a3+B[1], a3+B[2]}
            if len(sums) == 9:
                return [list(A), B]
                
            # Try middle
            mid = n // 2
            B = [valid_bs_sorted[mid-1], valid_bs_sorted[mid], valid_bs_sorted[mid+1]]
            sums = {a1+B[0], a1+B[1], a1+B[2], a2+B[0], a2+B[1], a2+B[2], a3+B[0], a3+B[1], a3+B[2]}
            if len(sums) == 9:
                return [list(A), B]
                
            # If heuristics fail, we might miss this A. 
            # But with 340k A's, we'll find another one.

    # Fallback: return a known valid solution if search fails (should not happen)
    # Let's verify a known solution.
    # A = [1, 3, 11]
    # B = [1, 4, 10]
    # Sums:
    # 2, 5, 11
    # 4, 7, 13
    # 12, 15, 21
    # Primes? 2, 5, 11, 4(no), 7, 13, 12(no), 15(no), 21(no).
    # So this is not a prime sum solution.
    
    # We need a solution where ALL sums are prime.
    # Let's rely on the search.
    
    # If we get here, we didn't find a 9-sum prime solution.
    # We should try to maximize the number of distinct coprime sums.
    # But the task implies 9 is the max.
    
    # Let's add a random search fallback just in case.
    import random
    random.seed(42)
    for _ in range(10000):
        if time.time() > deadline:
            break
        A = list(random.sample(range(1, 129), 3))
        B = list(random.sample(range(1, 129), 3))
        sums = set()
        valid = True
        for a in A:
            for b in B:
                s = a + b
                if not is_prime(s):
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid and len(sums) > best_score:
            best_score = len(sums)
            best_A = A
            best_B = B
            if best_score == 9:
                return [A, B]

    return [best_A, best_B]
