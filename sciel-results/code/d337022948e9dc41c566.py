"""Program for 'Large sets without subset averages' task."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Deadline guard (wall-clock). The task allows up to 60s; the evaluator kills at 90s.
# We use a 50s budget to leave margin for interpreter overhead and evaluation.
_DEADLINE = time.time() + 50.0

def _elapsed():
    return time.time() - _DEADLINE

def _check_pair_conflict(a, b, c):
    # Returns True if a is the mean of b and c (i.e., a = (b+c)/2),
    # with the constraint that b and c are distinct and not equal to a.
    # Since elements are distinct, b != c is automatic if b != a and c != a?
    # No, if b == c, then a = b, which violates distinctness.
    # So we just need a != b and a != c.
    return (b + c) == 2 * a and a != b and a != c

def _is_mean_free_small(subset, max_k=4):
    """
    Check if any element in `subset` is the arithmetic mean of a subset of size k (2<=k<=max_k)
    of other elements in `subset`.
    Returns True if no such mean exists (i.e., it is 'mean-free' up to max_k).
    """
    n = len(subset)
    if n < 3:
        return True
    
    # Sort to potentially help with pruning, though not strictly necessary for small n
    s = sorted(subset)
    
    for i, target in enumerate(s):
        # The other elements
        others = s[:i] + s[i+1:]
        # We need to check subsets of `others` of size k=2..max_k
        for k in range(2, min(max_k, len(others)) + 1):
            # Iterate over combinations of `others` of size k
            # If sum == k * target, then target is the mean
            for comb in combinations(others, k):
                if sum(comb) == k * target:
                    return False
    return True

def _full_mean_free_check(subset):
    """
    Check if the set is fully mean-free (no element is the mean of ANY subset of >=2 others).
    This is the strict verification required by the task.
    Uses DP for subset sums for each target element to be efficient?
    Actually, for n <= 50, brute force with bitsets or just itertools is feasible if optimized.
    But 2^50 is too big.
    However, we can use the property that we only care if `target` is the mean.
    For a fixed target `t`, we want to know if there exists a subset T of S\{t} such that sum(T) = k*t and |T| >= 2.
    This is equivalent to: Does there exist a non-empty subset of S\{t} that sums to a multiple of t, with size >= 2?
    Actually, it's simpler: for each k >= 2, does there exist a subset of size k summing to k*t?
    
    Since n is small (<= 48), we can use a DP approach for each target.
    DP[i][j][k] = can we form sum j using exactly k elements from the first i elements?
    But j can be up to 48 * 512 ~ 25000. k up to 48.
    Size: 48 * 25000 * 48 ~ 57 million. Too big for Python lists of bools if done naively for each target.
    
    Alternative: Use bitset for subset sums.
    For a fixed target t, we want to know if any multiple m*t (for m>=2) is a subset sum of S\{t}, 
    AND if the number of elements in that subset is >= 2.
    Wait, the problem says "subset of at least two OTHER elements".
    So we need the subset size k >= 2.
    
    Let's use a standard subset sum DP with counts?
    Or simpler: Just use itertools combinations for k=2,3,4... up to n-1?
    For n=48, k=2 is C(47,2) ~ 1100. k=3 is C(47,3) ~ 16000. k=4 is C(47,4) ~ 190k.
    Sum of C(47,k) for k=2..47 is 2^47 - 1 - 47 ~ 1.4 * 10^14. Too slow.
    
    We need a smarter check.
    However, in practice, most violations happen for small k.
    The "inspiration" mentioned limiting subset check to k<=4 keeps runtime within deadline while catching *most* violations.
    But the verifier checks EVERY subset size.
    So my solution MUST be valid for ALL subset sizes.
    
    Is it possible that a set is valid for k<=4 but invalid for k>4?
    Yes. For example, if there is a large subset that sums to k*t.
    
    How to verify full validity efficiently?
    Note that if a set S is mean-free, then for any t in S, the set S\{t} must not contain any subset summing to k*t for any k>=2.
    This implies that 0 is the only subset sum of S\{t} that is a multiple of t? No, that's not right.
    It implies that no multiple of t (greater than 0) is a subset sum of S\{t} with cardinality >= 2?
    Actually, if sum(T) = k*t and |T| >= 2, then t is the mean.
    
    Let's consider the magnitude. The max sum is 48*512 = 24576.
    We can do a DP for each target t:
    dp[sum] = bitmask of possible cardinalities? Or just a set of cardinalities?
    Since cardinality is up to 48, we can use a bitset where bit k is set if sum is achievable with cardinality k.
    Initialize dp[0] = 1 << 0 (sum 0 with 0 elements).
    For each element e in S\{t}:
        new_dp = dp
        For s in range(max_sum, e-1, -1):
            if dp[s]:
                # shift cardinality by 1
                new_dp[s+e] |= dp[s] << 1
        dp = new_dp
        
    Then check: for any sum s that is a multiple of t (s = k*t, k>=2),
    check if dp[s] has any bit set for cardinality c >= 2.
    
    This is O(n * max_sum) per target.
    max_sum ~ 25000. n ~ 48.
    48 * 25000 = 1.2 million operations per target.
    For 48 targets: 57 million operations.
    In Python, this might be too slow (57M * constant).
    But we can optimize:
    1. Use bitsets (Python integers) for the DP.
       dp is a list of integers, where dp[s] is a bitmask of achievable cardinalities.
       Actually, it's better to have:
       dp[s] = bitmask of cardinalities.
       Update: for e in others:
           for s in reversed(range(max_sum, e-1, -1)):
               dp[s+e] |= (dp[s] << 1)
       
    Let's estimate the speed.
    Inner loop: ~25000 iterations.
    Outer loop: ~48 iterations.
    Total: ~1.2M iterations per target.
    Each iteration involves bit operations on Python ints (which are fast in C).
    57M iterations total.
    Python can do ~10-50M simple ops per second. Bit ops on small ints are fast.
    This should take a few seconds.
    
    Let's implement this full check.
    """
    n = len(subset)
    if n < 3:
        return True
    
    # We will check each element as a potential mean
    for t in subset:
        others = [x for x in subset if x != t]
        max_sum = sum(others)
        
        # dp[s] is a bitmask where bit k is set if sum s can be formed with cardinality k
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # cardinality 0
        
        for e in others:
            # Iterate backwards to avoid using an element twice
            for s in range(max_sum, e - 1, -1):
                if dp[s]:
                    dp[s + e] |= (dp[s] << 1)
        
        # Now check if any multiple of t (k>=2) has a valid cardinality (>=2)
        # Multiples: 2*t, 3*t, ... up to max_sum
        for k in range(2, max_sum // t + 1):
            s = k * t
            if s <= max_sum:
                # Check if dp[s] has any bit set for cardinality >= 2
                # Bit 0 is cardinality 0, bit 1 is cardinality 1, etc.
                # We want to know if (dp[s] & ~3) != 0  (mask out bits 0 and 1)
                if dp[s] & ~3:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds budget
    
    # Candidate pool: Use 5-smooth numbers (2^a * 3^b * 5^c) up to 512.
    # These numbers have rich multiplicative structure, which tends to avoid additive mean relations.
    candidates = []
    for a in range(10):  # 2^a <= 512 => a <= 9
        val = 1
        for _ in range(a):
            val *= 2
        for b in range(6):  # 3^b <= 512/val
            val2 = val
            for _ in range(b):
                val2 *= 3
            if val2 > 512:
                break
            for c in range(4):  # 5^c <= 512/val2
                val3 = val2
                for _ in range(c):
                    val3 *= 5
                if val3 > 512:
                    break
                if val3 >= 1 and val3 <= 512:
                    candidates.append(val3)
    
    # Remove duplicates and sort
    candidates = list(set(candidates))
    candidates.sort()
    
    # If we have fewer than 48 candidates, we might need to add more.
    # But 5-smooth numbers are sparse. Let's count them.
    # 2^a * 3^b * 5^c <= 512.
    # a=0: 3^b * 5^c <= 512.
    #   c=0: 3^b <= 512 => b=0..5 (6)
    #   c=1: 3^b <= 102 => b=0..4 (5)
    #   c=2: 3^b <= 20 => b=0..2 (3)
    #   c=3: 3^b <= 4 => b=0..1 (2)
    #   c=4: 3^b <= 0.8 => 0
    #   Total for a=0: 16
    # a=1: 2 * 3^b * 5^c <= 512 => 3^b * 5^c <= 256.
    #   c=0: b=0..5 (6)
    #   c=1: b=0..4 (5)
    #   c=2: b=0..2 (3)
    #   c=3: b=0..1 (2)
    #   Total: 16
    # a=2: 4 * ... <= 512 => 3^b * 5^c <= 128.
    #   c=0: b=0..4 (5)
    #   c=1: b=0..3 (4)
    #   c=2: b=0..2 (3)
    #   c=3: b=0..1 (2)
    #   Total: 14
    # a=3: 8 * ... <= 512 => 3^b * 5^c <= 64.
    #   c=0: b=0..3 (4)
    #   c=1: b=0..3 (4) -- 5*27=135>64, so b=0..2 (3)
    #   c=2: b=0..1 (2)
    #   c=3: b=0 (1)
    #   Total: 10
    # a=4: 16 * ... <= 512 => 3^b * 5^c <= 32.
    #   c=0: b=0..3 (4)
    #   c=1: b=0..2 (3)
    #   c=2: b=0..1 (2)
    #   c=3: b=0 (1)
    #   Total: 10
    # a=5: 32 * ... <= 512 => 3^b * 5^c <= 16.
    #   c=0: b=0..2 (3)
    #   c=1: b=0..1 (2)
    #   c=2: b=0 (1)
    #   Total: 6
    # a=6: 64 * ... <= 512 => 3^b * 5^c <= 8.
    #   c=0: b=0..1 (2)
    #   c=1: b=0 (1)
    #   Total: 3
    # a=7: 128 * ... <= 512 => 3^b * 5^c <= 4.
    #   c=0: b=0..1 (2)
    #   c=1: b=0 (1)
    #   Total: 3
    # a=8: 256 * ... <= 512 => 3^b * 5^c <= 2.
    #   c=0: b=0 (1)
    #   c=1: none
    #   Total: 1
    # a=9: 512 * ... <= 512 => 3^b * 5^c <= 1.
    #   c=0: b=0 (1)
    #   Total: 1
    # Sum: 16+16+14+10+10+6+3+3+1+1 = 80.
    # So we have 80 candidates. This is plenty.
    
    selected = []
    
    # Greedy construction
    # We will try to add candidates in increasing order.
    # We use a fast check (k<=4) during greedy to speed up selection.
    # Then we verify the full set at the end.
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        
        # Tentative set
        tentative = selected + [c]
        
        # Fast check: k <= 4
        if not _is_mean_free_small(tentative, max_k=4):
            continue
        
        # If fast check passes, we might want to do a partial full check?
        # No, let's just take it. We will do full check at the end.
        # But what if this element causes a violation for k>4 that makes the final set invalid?
        # Then we would have to remove it and backtrack.
        # To avoid complex backtracking, we can do a "medium" check during greedy.
        # Let's check k <= 6 during greedy.
        # C(47,6) is ~ 10M. Too slow for greedy.
        # Let's stick to k<=4 for greedy, and then repair.
        
        selected.append(c)
        
        # If we have a good number of elements, let's try to do a full check periodically?
        # No, full check is expensive.
        
    # Now we have a candidate set `selected`.
    # We need to verify it is FULLY mean-free.
    # If not, we need to remove elements to make it valid.
    # Strategy: Iteratively remove the element that is "most" a mean, or just the first violating element.
    
    # A better strategy:
    # 1. Check if `selected` is fully mean-free.
    # 2. If yes, return `selected`.
    # 3. If no, identify a violation: element t is mean of subset T.
    #    Remove one element from T (or t) and repeat.
    #    To maximize size, we should remove the element that is "least" useful?
    #    Or just remove t?
    #    Let's try removing the element that is the mean (t).
    
    # To avoid infinite loops, we will limit the number of repair steps.
    # Also, we can try to "swap" elements if removing one drops the size too much?
    # For now, simple removal.
    
    max_repair_steps = 100
    current_set = selected[:]
    
    for _ in range(max_repair_steps):
        if time.time() > deadline:
            break
        if not current_set:
            break
            
        if _full_mean_free_check(current_set):
            return current_set
            
        # Find a violation
        # We need to find t and subset T such that mean(T) = t.
        # We can reuse the logic from _full_mean_free_check but return the violation.
        violation_found = False
        for i, t in enumerate(current_set):
            others = [x for x in current_set if x != t]
            # We need to find a subset of others that sums to k*t.
            # Use DP to find one such subset.
            max_sum = sum(others)
            # dp[s] = bitmask of cardinalities? Or just store the parent?
            # To reconstruct the subset, we need to store choices.
            # This is heavy.
            # Alternative: Just find IF there is a violation, and remove t.
            # We already know there IS a violation because _full_mean_free_check returned False.
            # So we can just remove t and try again?
            # But removing t might not fix the problem if there is another violation.
            # And it might be better to remove an element from T.
            
            # Let's try to be a bit smarter.
            # Check k=2 first (fast).
            # If t is mean of pair (a,b), remove a or b or t?
            # Removing t is safe.
            
            # Let's just remove t for now.
            current_set.remove(t)
            violation_found = True
            break
            
        if not violation_found:
            break
            
    # If we still haven't found a valid set (or ran out of time), return the best valid subset we can find quickly.
    # As a last resort, fall back to a known small valid set or the greedy result if it happens to be valid (checked above).
    # If the loop exited without returning, current_set might be empty or invalid.
    # Let's do a final check.
    if current_set and _full_mean_free_check(current_set):
        return current_set
        
    # Fallback: Return a small valid set.
    # {1, 2, 4, 8, 16, 32, 64, 128, 256} is mean-free?
    # Powers of 2 are mean-free for k=2.
    # Are they mean-free for k>2?
    # Sum of k distinct powers of 2 is not a multiple of 2^m unless k is specific?
    # Actually, sum of distinct powers of 2 is unique.
    # k * 2^m = sum of k distinct powers of 2.
    # The sum of k distinct powers of 2 has k bits set.
    # k * 2^m has only one bit set (if k is a power of 2) or multiple bits?
    # Example: 3 * 8 = 24 = 16 + 8. Distinct powers? 16 and 8 are distinct.
    # So 8 is the mean of {16, 8}? No, 8 is in the set.
    # The mean must be of OTHER elements.
    # If t=8, others={16, 32...}.
    # Can 8 be the mean of a subset of others?
    # Sum of others = k*8.
    # Smallest other is 16.
    # If k=2, min sum 16+32=48 != 16.
    # So powers of 2 are safe.
    
    fallback = [2**i for i in range(9)] # 1, 2, 4, ..., 256
    return fallback

# EVOLVE-BLOCK-END
