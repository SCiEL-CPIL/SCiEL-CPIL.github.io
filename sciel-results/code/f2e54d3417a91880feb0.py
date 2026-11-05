"""Rewritten program: exhaustive search with prime-mask pruning for pairwise-coprime sumsets.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i+b_j the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage odd prime factors.

Search strategy:
  1. Precompute smallest prime factor (SPF) up to 256.
  2. For each odd integer s in [3, 255], compute a bitmask of odd prime divisors.
     There are 61 odd primes <= 256 (primes 3,5,7,...,251).
  3. Enumerate all (A, B) where A is 3 distinct odd numbers in [1,128] and B is 3 distinct
     even numbers in [1,128] (and the symmetric case).
  4. For each candidate, compute the 9 sums, get their masks, and check pairwise coprime
     by verifying that for every pair of distinct sums, mask_i & mask_j == 0.
  5. Track the maximum number of distinct sums achieved.

To make this feasible, we use combinatorial enumeration with early pruning:
  - Number of odd numbers in [1,128]: 64
  - Number of even numbers in [1,128]: 64
  - C(64,3) = 41664 for each set, so total pairs ~ 41664^2 ≈ 1.7e9 which is too many.

  Better approach: iterate over all C(64,3) * C(64,3) but with strong early termination.
  Actually, let's be smarter. We can iterate over A first, then for each A iterate over B,
  but prune B early if any sum conflicts.

  Alternative: since we want to MAXIMIZE distinct sums (max 9), we can check if 9 is achievable.
  If not, check 8, etc. But it's easier to just do a full search with good pruning.

  Let's estimate: 41664 * 41664 ≈ 1.74e9 pairs. Each pair check involves 9 sums, each mask
  computation is O(1) (precomputed), and pairwise coprime check is O(9^2) = 81 bit-and operations.
  That's about 1.74e9 * 81 ≈ 1.4e11 operations. Too slow in Python.

  Need better pruning. Key insight: for a fixed A, the sums a_i + b must have masks that are
  pairwise disjoint across all pairs. Let's precompute for each possible b, the mask of a_i+b
  for all i. Then for a fixed A, we need to find 3 distinct b's such that the union of masks
  across all 9 sums has no overlapping primes between any two distinct sums.

  Actually, let's just do a smarter enumeration. We'll iterate over all A (41664 of them), and
  for each A, iterate over all B (41664 of them), but with early exit. In Python this might
  still be too slow.

  Let me think of a different approach. Since the answer is likely small (maybe 5-7 distinct sums),
  and we just need to find the maximum, we can use a randomized or heuristic search, or we can
  reduce the search space.

  Actually, let me just try the brute force with NumPy vectorization where possible, or use
  a more clever enumeration.

  Let's try: for each A, compute all possible b values and their resulting masks. Then we need
  to pick 3 b's such that all 9 sums have pairwise disjoint masks. This is a constraint satisfaction
  problem.

  Given time constraints, let me just do a direct search with early pruning and hope it's fast
  enough, or use a smarter method.

  Let me try a different tactic: enumerate all possible sets of distinct odd sums that are
  pairwise coprime, then check if they can be represented as a 3x3 sumset.

  Actually, the simplest correct approach: just brute force with early termination.
  Let's optimize the inner loop heavily.
"""

import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, with 5s safety margin
    
    # Precompute smallest prime factor
    MAX_VAL = 256
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Get all odd primes up to 256
    odd_primes = [p for p in range(3, MAX_VAL + 1) if spf[p] == p]
    prime_to_idx = {p: i for i, p in enumerate(odd_primes)}
    num_odd_primes = len(odd_primes)
    
    # For each odd number s in [1, 255], compute bitmask of odd prime divisors
    def get_mask(s):
        if s <= 1:
            return 0
        mask = 0
        x = s
        while x > 1:
            p = spf[x]
            if p != 2:
                mask |= (1 << prime_to_idx[p])
            while x % p == 0:
                x //= p
        return mask
    
    # Precompute masks for all odd numbers 1..255
    masks = [0] * (MAX_VAL + 1)
    for s in range(1, MAX_VAL + 1, 2):
        masks[s] = get_mask(s)
    
    # Odd numbers in [1, 128]
    odds = [i for i in range(1, 129) if i % 2 == 1]
    evens = [i for i in range(2, 129) if i % 2 == 0]
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Function to check if a set of sums is pairwise coprime
    def check_pairwise_coprime(sums):
        """Check if distinct values in sums are pairwise coprime."""
        distinct_sums = list(set(sums))
        n = len(distinct_sums)
        for i in range(n):
            mi = masks[distinct_sums[i]]
            for j in range(i + 1, n):
                if mi & masks[distinct_sums[j]]:
                    return False
        return True
    
    # Search over A (odd) and B (even)
    # To keep it feasible, we'll iterate and check time
    all_odds = odds  # 64 elements
    all_evens = evens  # 64 elements
    
    # Enumerate all combinations of 3 from odds and evens
    # Use early pruning
    count_A = 0
    for A in combinations(all_odds, 3):
        if time.time() - start_time > DEADLINE:
            break
        count_A += 1
        a0, a1, a2 = A
        for B in combinations(all_evens, 3):
            if time.time() - start_time > DEADLINE:
                break
            b0, b1, b2 = B
            sums = [a0+b0, a0+b1, a0+b2, a1+b0, a1+b1, a1+b2, a2+b0, a2+b1, a2+b2]
            distinct_sums = list(set(sums))
            cnt = len(distinct_sums)
            if cnt <= best_count:
                continue
            # Check pairwise coprime
            if check_pairwise_coprime(distinct_sums):
                best_count = cnt
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]
    
    # Also search the symmetric case: A even, B odd
    if time.time() - start_time < DEADLINE:
        for A in combinations(all_evens, 3):
            if time.time() - start_time > DEADLINE:
                break
            a0, a1, a2 = A
            for B in combinations(all_odds, 3):
                if time.time() - start_time > DEADLINE:
                    break
                b0, b1, b2 = B
                sums = [a0+b0, a0+b1, a0+b2, a1+b0, a1+b1, a1+b2, a2+b0, a2+b1, a2+b2]
                distinct_sums = list(set(sums))
                cnt = len(distinct_sums)
                if cnt <= best_count:
                    continue
                if check_pairwise_coprime(distinct_sums):
                    best_count = cnt
                    best_A = list(A)
                    best_B = list(B)
                    if best_count == 9:
                        return [best_A, best_B]
    
    return [best_A, best_B]
