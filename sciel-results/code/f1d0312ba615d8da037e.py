import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Key insight: if we use a geometric sequence with ratio r > 1,
    # then the minimum sum of a (k+1)-subset is r^k (1 + r + ... + r^k) * min_element
    # and the maximum sum of a k-subset is (r^k + r^{k+1} + ... + r^{2k-1}) * min_element
    # 
    # Actually, for powers of 2: 1, 2, 4, 8, ...
    # Sum of k elements from {1, 2, 4, ..., 2^{n-1}} ranges from:
    #   min = 1 + 2 + ... + 2^{k-1} = 2^k - 1
    #   max = 2^{n-k} + 2^{n-k+1} + ... + 2^{n-1} = 2^n - 2^{n-k}
    #
    # For k and k+1 to be disjoint, we need:
    #   max sum of k-subset < min sum of (k+1)-subset
    #   2^n - 2^{n-k} < 2^{k+1} - 1
    #
    # This is very restrictive. Let's try a different approach.
    #
    # Better insight: Use a sequence where each element is larger than the sum of all previous.
    # i.e., a_i > sum(a_1, ..., a_{i-1})
    # Then subset sums are unique (like binary representation).
    # But we need cardinality-separated, not just unique.
    #
    # If a_i > sum of all previous, then:
    # - The minimum (k+1)-subset sum includes a_k (the k-th smallest) plus k elements from {a_1,...,a_{k-1}}
    #   Wait, that's not right. Let me reconsider.
    #
    # Let's use: a_1 < a_2 < ... < a_n, and a_{k} > sum_{i=1}^{k-1} a_i
    # Then any subset of size k has sum in a specific range.
    #
    # Actually, the simplest valid set is powers of 2: {1, 2, 4, 8, ..., 2^{n-1}}
    # But the maximum element must be <= 256, so 2^{n-1} <= 256, n-1 <= 8, n <= 9.
    # That gives only 9 elements. Score = 9/seed.
    #
    # We need a denser set. Let's think about the constraint more carefully.
    #
    # Constraint: For any two subsets A, B with |A| != |B|, sum(A) != sum(B).
    #
    # This is equivalent to: the generating function P(x, y) = prod_{a in S} (1 + x y^a)
    # has no two terms with the same y-exponent but different x-exponents.
    #
    # Or: for each sum value s, at most one cardinality k can produce sum s.
    #
    # Let's try: S = {a_1, a_2, ..., a_n} where a_i are chosen so that
    # the k-th order "gap" is large enough.
    #
    # Alternative: Use a base-b representation idea.
    # If we pick elements as multiples of a large number plus small offsets...
    #
    # Let's try a constructive approach:
    # Pick a large base B. Let S = {B + 1, B + 2, ..., B + n}
    # Sum of k elements = k*B + (sum of k offsets from 1..n)
    # For two different cardinalities k and m (k < m):
    #   k*B + s_k = m*B + s_m  =>  (m-k)*B = s_k - s_m
    #   |s_k - s_m| <= n*n (max sum of n offsets is n*(n+1)/2, min is n*(n+1)/2 for the other)
    #   Actually max s_k = k*(k+1)/2, min s_m = m*(m+1)/2
    #   So |s_k - s_m| <= max(k*(k+1)/2, m*(m+1)/2)
    #   We need (m-k)*B > max possible |s_k - s_m|
    #   B > max_{k,m} max(k*(k+1)/2, m*(m+1)/2) / 1  (since m-k >= 1)
    #   B > n*(n+1)/2
    #
    # So if we pick B > n*(n+1)/2, and S = {B+1, ..., B+n}, the constraint is satisfied!
    # Max element = B + n <= 256, so B <= 256 - n
    # Need B > n*(n+1)/2, so 256 - n > n*(n+1)/2
    # 256 > n*(n+1)/2 + n = n(n+3)/2
    # n(n+3) < 512
    # Try n=20: 20*23=460 < 512. OK!
    # n=21: 21*24=504 < 512. OK!
    # n=22: 22*25=550 > 512. No.
    # So max n = 21 with this simple scheme.
    #
    # But we can do better! The offsets don't have to be 1..n.
    #
    # Let's generalize: S = {a_1, ..., a_n} with a_i = B + c_i, where c_i are small distinct integers.
    # Sum of k elements = k*B + sum of k c's.
    # For k != m: k*B + S_k = m*B + S_m => (m-k)*B = S_k - S_m
    # We need |S_k - S_m| < (m-k)*B for all k != m.
    # The worst case is m = k+1, so B > |S_k - S_{k+1}|
    # S_k is sum of k c's, S_{k+1} is sum of k+1 c's.
    # |S_k - S_{k+1}| <= max sum of k+1 c's - min sum of k c's
    # If c_i are in range [0, C], then |S_k - S_{k+1}| <= (k+1)*C
    # So B > (k+1)*C for all k, i.e., B > n*C
    # Max element = B + C <= 256, so B <= 256 - C
    # Need 256 - C > n*C => 256 > C(n+1) => C < 256/(n+1)
    # Also c_i are distinct integers in [0, C], so we need C+1 >= n, i.e., C >= n-1.
    # So we need n-1 <= C < 256/(n+1)
    # n-1 < 256/(n+1) => (n-1)(n+1) < 256 => n^2 - 1 < 256 => n^2 < 257 => n <= 16
    #
    # So this scheme gives at most 16 elements. Worse than the previous 21.
    #
    # Let's go back to S = {B+1, ..., B+n} with B > n(n+1)/2.
    # n=21 works. Can we push further?
    #
    # What if we don't use consecutive offsets? We can use a better set of offsets.
    #
    # Actually, let me reconsider. The constraint is that for each sum s, at most one cardinality produces it.
    # This is a stronger condition than what I was checking.
    #
    # Let me just use a greedy approach with the cardinality-indexed DP, but be smarter about element selection.
    #
    # Strategy: Start with a good seed (like the B+1..B+n scheme for n=21), then try to add more elements.
    
    # First, get a baseline valid set
    def is_valid_set(S):
        """Check if subset sums of different cardinalities are disjoint."""
        if not S:
            return True
        n = len(S)
        max_sum = sum(S)
        # K[k] = set of subset sums of cardinality k
        K = [set() for _ in range(n + 1)]
        K[0].add(0)
        for x in S:
            for k in range(n - 1, -1, -1):
                if K[k]:
                    new_sums = {s + x for s in K[k]}
                    K[k + 1].update(new_sums)
        # Check disjointness across cardinalities
        seen = {}
        for k in range(n + 1):
            for s in K[k]:
                if s in seen and seen[s] != k:
                    return False
                seen[s] = k
        return True

    def count_conflicts(S):
        """Count number of cross-cardinality sum collisions."""
        if not S:
            return 0
        n = len(S)
        K = [set() for _ in range(n + 1)]
        K[0].add(0)
        for x in S:
            for k in range(n - 1, -1, -1):
                if K[k]:
                    new_sums = {s + x for s in K[k]}
                    K[k + 1].update(new_sums)
        # Count collisions
        sum_to_card = {}
        conflicts = 0
        for k in range(n + 1):
            for s in K[k]:
                if s in sum_to_card:
                    if sum_to_card[s] != k:
                        conflicts += 1
                else:
                    sum_to_card[s] = k
        return conflicts

    # Baseline: use the B+1..B+n scheme
    # Find max n such that B+1..B+n with B > n(n+1)/2 and B+n <= 256
    best_n = 0
    for n in range(1, 22):
        min_B = n * (n + 1) // 2 + 1
        if min_B + n <= 256:
            best_n = n
        else:
            break
    
    baseline_n = best_n
    min_B = baseline_n * (baseline_n + 1) // 2 + 1
    S_baseline = list(range(min_B, min_B + baseline_n))
    
    # Verify baseline is valid
    if not is_valid_set(S_baseline):
        # Fallback to powers of 2
        S_baseline = [2**i for i in range(9)]  # 1,2,4,...,128
    
    current_S = S_baseline[:]
    current_conflicts = 0
    
    # Now try to add more elements using local search
    random.seed(42)
    
    # Try adding elements one by one, choosing the one that causes fewest conflicts
    used = set(current_S)
    
    for _ in range(200):  # Try up to 200 additions
        if time.time() > deadline - 5:
            break
        
        best_x = None
        best_conflict_increase = float('inf')
        
        candidates = [x for x in range(1, 257) if x not in used]
        
        # Sample candidates for speed
        if len(candidates) > 50:
            candidates = random.sample(candidates, 50)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            test_S = current_S + [x]
            c = count_conflicts(test_S)
            if c == 0:
                best_x = x
                best_conflict_increase = 0
                break
            elif c < best_conflict_increase:
                best_conflict_increase = c
                best_x = x
        
        if best_x is not None and best_conflict_increase == 0:
            current_S.append(best_x)
            used.add(best_x)
        
        if len(current_S) >= 60:
            break
    
    # If we have a valid set, return it
    if count_conflicts(current_S) == 0:
        return current_S
    
    # If not, fall back to the baseline
    return S_baseline

# Test
if __name__ == "__main__":
    s = construct()
    print(f"Set size: {len(s)}")
    print(f"Set: {s}")
    print(f"Valid: {is_valid_set(s)}")
